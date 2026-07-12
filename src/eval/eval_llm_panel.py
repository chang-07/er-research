"""Compare multiple LLM-baseline prediction files on the SAME anchor set
per split. Anchors not scored by all listed models are dropped from the
comparison so the row-by-row numbers are directly comparable.

Usage:
  python3 src/eval/eval_llm_panel.py \\
      --preds llm_haiku_panel llm_gptoss_panel llm_qwen3_panel
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def per_anchor_hits_and_rr(df):
    """Yield (anchor_lei, hit_at_1, reciprocal_rank) for eligible anchors."""
    for anchor, sub in df.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        first = int(pi[0]) + 1
        yield anchor, (1 if first == 1 else 0), 1.0 / first


def bootstrap_ci(values, n_boots=500, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    boots = np.empty(n_boots)
    for i in range(n_boots):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots[i] = arr[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--preds", nargs="+", required=True,
                    help="prediction file stems (without .parquet)")
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label",
                  "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()

    # Load each prediction file and remember its scored-anchor set
    models = {}
    for stem in args.preds:
        path = os.path.join(args.data, "predictions", f"{stem}.parquet")
        if not os.path.exists(path):
            print(f"  SKIP {stem}: file not found", file=sys.stderr)
            continue
        preds = pq.read_table(path).to_pandas()
        anchors = set(preds["anchor_lei"])
        models[stem] = (preds, anchors)
        print(f"  {stem}: {len(preds):,} pairs, {len(anchors):,} anchors",
              file=sys.stderr)

    if not models:
        print("no models loaded", file=sys.stderr); sys.exit(1)

    print()
    splits = ("split_time", "split_event", "split_jurisdiction")
    short = {"split_time": "TIME", "split_event": "EVENT",
              "split_jurisdiction": "JURIS"}

    for split in splits:
        # Anchors that ALL models FULLY scored on this split's test set.
        # "Fully" matters: an anchor with any unscored candidate (e.g. an
        # API error the runner excluded) would be ranked on a partial pool,
        # silently shifting Hits@1 for that model only.
        test_split = pairs[pairs[split] == "test"]
        common = None
        for stem, (preds, _) in models.items():
            merged = test_split.merge(
                preds[["anchor_lei", "candidate_lei", "score"]],
                on=["anchor_lei", "candidate_lei"], how="left")
            ok = merged.groupby("anchor_lei")["score"].apply(
                lambda s: s.notna().all())
            s_anchors = set(ok[ok].index)
            common = s_anchors if common is None else (common & s_anchors)
        common = common or set()
        print(f"=== {short[split]} ({len(common)} anchors scored by all models) ===")
        print(f"{'model':<25s}  {'H@1':>6s} [{'lo':>5s},{'hi':>5s}]  "
              f"{'MRR':>6s} [{'lo':>5s},{'hi':>5s}]")

        for stem, (preds, _) in models.items():
            df = test_split[test_split["anchor_lei"].isin(common)].merge(
                preds[["anchor_lei", "candidate_lei", "score"]],
                on=["anchor_lei", "candidate_lei"], how="left")
            df = df[df["score"].notna()].copy()
            hits, rrs = [], []
            for _, h, r in per_anchor_hits_and_rr(df):
                hits.append(h); rrs.append(r)
            if not hits:
                continue
            h_mean = float(np.mean(hits))
            r_mean = float(np.mean(rrs))
            h_lo, h_hi = bootstrap_ci(hits)
            r_lo, r_hi = bootstrap_ci(rrs)
            print(f"{stem:<25s}  {h_mean:>6.4f} [{h_lo:>5.3f},{h_hi:>5.3f}]  "
                  f"{r_mean:>6.4f} [{r_lo:>5.3f},{r_hi:>5.3f}]")
        print()


if __name__ == "__main__":
    main()
