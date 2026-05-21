"""Compare model performance across hard-negative distributions.

Tests whether cross-encoder gains are robust to the choice of hard-negative
miner. The original paper trains on v2 (E5-cosine) negs and evaluates on a
pairs file built with v1 (token-overlap) negs. This script evaluates the
*same* metric pipeline on two evaluation distributions:

  v2 candidate pool:  data/pairs.parquet         (v1 token-overlap negs)
  v3 candidate pool:  data/pairs_v3.parquet      (TF-IDF char-3gram negs)

For each (model, eval-distribution) cell we report:
  - Hits@1 with bootstrap 95% CI
  - AUROC
  - Delta vs the best zero-shot baseline on the SAME distribution
    (so the "miner artifact" question is answered by comparing deltas)

Usage:
  python3 src/eval/compare_distributions.py --data data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

SPLIT_REGIMES = ["split_time", "split_event", "split_jurisdiction"]

# (preds_filename, label) tuples — keep "_baseline" or "_cross" for grouping
EVALS = {
    "pairs.parquet": [
        ("tfidf_char",                  "tfidf_char         (v2 pool)",      "baseline"),
        ("sentence_encoder",            "E5-small zero-shot (v2 pool)",      "baseline"),
        ("method_v2_full",              "bi-encoder v2_full (v2 pool)",      "bi-encoder"),
        ("method_v2_crossenc",          "cross-enc trn-on-v2 (v2 pool, ORIGINAL)", "cross"),
        ("method_v2_crossenc_repro",    "cross-enc trn-on-v2 (v2 pool, repro)",    "cross"),
        ("method_v3_crossenc_on_pairs", "cross-enc trn-on-v3 (v2 pool)",     "cross"),
    ],
    "pairs_v3.parquet": [
        ("tfidf_char_v3",                            "tfidf_char         (v3 pool)", "baseline"),
        ("sentence_encoder_v3",                      "E5-small zero-shot (v3 pool)", "baseline"),
        ("method_v2_crossenc_repro_on_pairs_v3",     "cross-enc trn-on-v2 (v3 pool)", "cross"),
        ("method_v3_crossenc",                       "cross-enc trn-on-v3 (v3 pool)", "cross"),
    ],
}


def hits1(d):
    """Per-anchor Hits@1, matching evaluate.py logic exactly."""
    hits, n = 0, 0
    for _, sub in d.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pos_idx = ranked.index[ranked["label"] == 1]
        if len(pos_idx) == 0:
            continue
        first = int(pos_idx[0]) + 1
        hits += (1 if first == 1 else 0)
        n += 1
    return (hits / n, n) if n > 0 else (0.0, 0)


def bootstrap_hits1_ci(d, n_boots=150, seed=42):
    has_pos = d.groupby("anchor_lei")["label"].max()
    valid = has_pos[has_pos == 1].index.to_numpy()
    if len(valid) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boots):
        sample = rng.choice(valid, size=len(valid), replace=True)
        sub = d[d.anchor_lei.isin(sample)]
        h, _ = hits1(sub)
        boots.append(h)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-boots", type=int, default=150)
    ap.add_argument("--skip-ci", action="store_true",
                    help="skip bootstrap CIs (faster)")
    args = ap.parse_args()

    results = []  # rows: (eval_dist, model_label, kind, split, hits1, ci_lo, ci_hi, auroc, n)

    for pairs_file, model_list in EVALS.items():
        path = os.path.join(args.data, pairs_file)
        if not os.path.exists(path):
            print(f"  skip eval-distribution {pairs_file} (file not found)",
                  file=sys.stderr)
            continue
        print(f"\n=== eval distribution: {pairs_file} ===", file=sys.stderr)
        pairs = pq.read_table(path,
                               columns=["anchor_lei", "candidate_lei", "label",
                                        "split_time", "split_event",
                                        "split_jurisdiction"]).to_pandas()
        print(f"  {len(pairs):,} pair rows", file=sys.stderr)

        for pred_stem, model_label, kind in model_list:
            pred_path = os.path.join(args.data, "predictions",
                                      f"{pred_stem}.parquet")
            if not os.path.exists(pred_path):
                print(f"  skip {model_label} ({pred_stem}.parquet missing)",
                      file=sys.stderr)
                continue
            preds = pq.read_table(pred_path).to_pandas()
            df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"],
                              how="left")
            miss = df["score"].isna().sum()
            if miss:
                df["score"] = df["score"].fillna(0.0)

            for split_col in SPLIT_REGIMES:
                test = df[df[split_col] == "test"]
                if len(test) == 0 or test["label"].nunique() < 2:
                    continue
                h1, n = hits1(test)
                ci_lo, ci_hi = (np.nan, np.nan)
                if not args.skip_ci:
                    ci_lo, ci_hi = bootstrap_hits1_ci(
                        test, n_boots=args.n_boots, seed=42)
                try:
                    auroc = roc_auc_score(test["label"].to_numpy(),
                                            test["score"].to_numpy())
                except ValueError:
                    auroc = float("nan")
                results.append({
                    "eval_dist": pairs_file,
                    "model_label": model_label,
                    "kind": kind,
                    "split": split_col,
                    "hits1": h1,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "auroc": auroc,
                    "n_anchors": n,
                })

    df = pd.DataFrame(results)
    if df.empty:
        print("no results — check prediction files exist", file=sys.stderr)
        sys.exit(1)

    # save raw
    out_csv = os.path.join(args.data, "compare_distributions.csv")
    df.to_csv(out_csv, index=False)

    # Pretty print: for each split, side-by-side table
    short_split = {"split_time": "TIME", "split_event": "EVENT",
                   "split_jurisdiction": "JURIS"}
    print()
    print("=" * 100)
    print("HITS@1 BY (MODEL, EVAL DISTRIBUTION, SPLIT)")
    print("=" * 100)
    for split_col in SPLIT_REGIMES:
        sub = df[df.split == split_col]
        if sub.empty:
            continue
        print(f"\n  -- {short_split[split_col]} split --")
        for _, r in sub.iterrows():
            ci_str = (f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                      if not np.isnan(r.ci_lo) else "")
            print(f"    {r.model_label:50s}  H@1={r.hits1:.3f} {ci_str:18s}  "
                  f"AUROC={r.auroc:.3f}  n={r.n_anchors}")

    # Key deltas: cross-encoder gain over best baseline, per distribution × split
    print()
    print("=" * 100)
    print("KEY DELTAS: cross-encoder gain over best zero-shot baseline")
    print("(if v2-trained cross-encoder generalizes, deltas should be comparable)")
    print("=" * 100)
    for split_col in SPLIT_REGIMES:
        print(f"\n  -- {short_split[split_col]} split --")
        for eval_dist in EVALS.keys():
            sub = df[(df.split == split_col) & (df.eval_dist == eval_dist)]
            if sub.empty:
                continue
            base = sub[sub.kind == "baseline"]
            cross = sub[sub.kind == "cross"]
            if base.empty or cross.empty:
                continue
            best_base = base.loc[base.hits1.idxmax()]
            print(f"    eval={eval_dist}: best baseline = "
                  f"{best_base.model_label} ({best_base.hits1:.3f})")
            for _, r in cross.iterrows():
                delta = r.hits1 - best_base.hits1
                print(f"      cross: {r.model_label:50s}  H@1={r.hits1:.3f}  "
                      f"Δ={delta:+.3f}  ({delta*100:+.1f}pp)")

    print(f"\nfull csv -> {out_csv}")


if __name__ == "__main__":
    main()
