"""Evaluate a sampled LLM prediction file (from llm_pairwise.py with
--test-anchors-per-split) on TIME / EVENT / JURIS test splits.

Standard evaluate.py joins the prediction file with pairs.parquet
and fills missing scores with 0, which would incorrectly penalise
unscored anchors. Here we restrict evaluation to the SAMPLED
anchors only (those for which the LLM scored every candidate pair)
and compute Hits@1 with bootstrap CIs on that subset.

Reports the same shape as the headline cross-encoder so the rows
slot into Table 1 directly.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def hits1_mrr_with_ci(df, n_boots=500, seed=42):
    """Return (Hits@1 mean+CI, MRR mean+CI, n_anchors)."""
    rng = np.random.default_rng(seed)
    hits, ranks = [], []
    for anchor, sub in df.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        first = int(pi[0]) + 1
        hits.append(1 if first == 1 else 0)
        ranks.append(1.0 / first)
    if not hits:
        return (float("nan"),) * 6 + (0,)
    h = np.asarray(hits, dtype=float)
    m = np.asarray(ranks, dtype=float)
    bh = np.empty(n_boots); bm = np.empty(n_boots)
    for i in range(n_boots):
        idx = rng.integers(0, len(h), size=len(h))
        bh[i] = h[idx].mean()
        bm[i] = m[idx].mean()
    return (float(h.mean()),
            float(np.percentile(bh, 2.5)),
            float(np.percentile(bh, 97.5)),
            float(m.mean()),
            float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)),
            int(len(h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pred", default="llm_haiku_panel",
                    help="prediction file stem under data/predictions/")
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label",
                  "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()

    pred_path = os.path.join(args.data, "predictions", f"{args.pred}.parquet")
    preds = pq.read_table(pred_path).to_pandas()
    scored_anchors = set(preds["anchor_lei"])
    print(f"loaded {len(preds):,} predictions across "
          f"{len(scored_anchors):,} anchors", file=sys.stderr)

    df = pairs.merge(preds[["anchor_lei", "candidate_lei", "score"]],
                      on=["anchor_lei", "candidate_lei"], how="left")

    print()
    print(f"{'split':<25s}  {'n_anc':>5s}  {'H@1':>6s} [{'lo':>5s},{'hi':>5s}]  "
          f"{'MRR':>6s} [{'lo':>5s},{'hi':>5s}]")
    print("-" * 80)
    for split in ("split_time", "split_event", "split_jurisdiction"):
        test = df[(df[split] == "test") & (df["anchor_lei"].isin(scored_anchors))]
        test = test[test["score"].notna()].copy()
        h, hlo, hhi, m, mlo, mhi, n = hits1_mrr_with_ci(test)
        print(f"{split:<25s}  {n:>5d}  "
              f"{h:>6.4f} [{hlo:>5.3f},{hhi:>5.3f}]  "
              f"{m:>6.4f} [{mlo:>5.3f},{mhi:>5.3f}]")
    print()


if __name__ == "__main__":
    main()
