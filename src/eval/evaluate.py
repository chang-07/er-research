"""Evaluate baselines on Sylvester v1.

Reads everything in data/predictions/ and joins against data/pairs.parquet
for labels and split assignments. For each (baseline x split_regime) it
computes:

  * AUROC, AUPRC  (threshold-free, on the test subset)
  * P / R / F1 @ best train-tuned threshold (held-out test subset)
  * Mean score per (label=0 negative class) — diagnostic for "which kind of
    negative tricks this baseline"

Outputs:
  data/eval_metrics.csv         long-format: baseline, split, metric, value
  data/eval_neg_breakdown.csv   long-format: baseline, neg_class, mean_score

Prints a compact pivoted table to stdout.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

SPLIT_REGIMES = ("split_time", "split_event", "split_jurisdiction")

def best_threshold(y_true, y_score):
    """Pick threshold maximizing F1 over the precision-recall curve."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra point; align by trimming
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / np.where(p + r > 0, p + r, 1)
    if len(t) == 0:
        return 0.5, 0.0
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])

def prf1(y_true, y_score, thr):
    pred = (y_score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def _per_anchor_ranks(df_test):
    """Return arrays parallel to anchors_with_pos: (anchor_lei, rank_of_first_pos).
    """
    anchors, ranks = [], []
    for anchor, sub in df_test.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pos_idx = ranked.index[ranked["label"] == 1]
        if len(pos_idx) == 0:
            continue
        anchors.append(anchor)
        ranks.append(int(pos_idx[0]) + 1)
    return np.asarray(anchors), np.asarray(ranks)


def retrieval_metrics(df_test):
    """Per-anchor retrieval: rank the true successor among candidates.

    For each anchor LEI with at least one positive, sort its candidates by
    score (desc) and look at the position of the positive. Returns mean
    Hits@1, Hits@3, MRR over anchors with a positive.
    """
    _, ranks = _per_anchor_ranks(df_test)
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0, 0
    h1 = float((ranks == 1).mean())
    h3 = float((ranks <= 3).mean())
    mrr = float((1.0 / ranks).mean())
    return h1, h3, mrr, int(len(ranks))


def bootstrap_hits1_ci(df_test, n_boots=200, seed=42):
    """Bootstrap 95% CI on Hits@1 by resampling anchors (with their full candidate
    sets) with replacement. Returns (lo, hi). Falls back to (nan, nan) if no
    anchors with positives are present.
    """
    anchors, ranks = _per_anchor_ranks(df_test)
    if len(anchors) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(anchors)
    boots = np.empty(n_boots, dtype=float)
    is_hit = (ranks == 1).astype(np.int8)
    for i in range(n_boots):
        idx = rng.integers(0, n, size=n)
        boots[i] = is_hit[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label", "source",
                 "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()
    print(f"loaded {len(pairs):,} pairs", file=sys.stderr)

    pred_files = sorted(glob.glob(os.path.join(args.data, "predictions", "*.parquet")))
    if not pred_files:
        print("no predictions found in data/predictions/", file=sys.stderr)
        sys.exit(1)

    metric_rows = []
    neg_rows = []

    for pf in pred_files:
        baseline = os.path.basename(pf).replace(".parquet", "")
        # Skip non-Sylvester prediction files (e.g., Magellan transfer
        # benchmarks that don't share the anchor_lei/candidate_lei schema).
        schema_cols = set(pq.read_schema(pf).names)
        if not {"anchor_lei", "candidate_lei", "score"}.issubset(schema_cols):
            print(f"  SKIP {baseline}: schema doesn't match", file=sys.stderr)
            continue
        preds = pq.read_table(pf, columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
        df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
        miss = df["score"].isna().sum()
        if miss:
            print(f"  WARN {baseline}: {miss:,} pairs missing scores; filling 0", file=sys.stderr)
            df["score"] = df["score"].fillna(0.0)

        for regime in SPLIT_REGIMES:
            train = df[df[regime] == "train"]
            test = df[df[regime] == "test"]
            if len(train) == 0 or len(test) == 0:
                continue
            if test["label"].nunique() < 2:
                continue
            y_train = train["label"].to_numpy()
            s_train = train["score"].to_numpy()
            y_test = test["label"].to_numpy()
            s_test = test["score"].to_numpy()

            auroc = roc_auc_score(y_test, s_test)
            auprc = average_precision_score(y_test, s_test)
            thr, _ = best_threshold(y_train, s_train)
            p, r, f1 = prf1(y_test, s_test, thr)
            h1, h3, mrr, n_anchors = retrieval_metrics(test)
            h1_lo, h1_hi = bootstrap_hits1_ci(test, n_boots=200, seed=42)

            metric_rows.extend([
                {"baseline": baseline, "split": regime, "metric": "auroc",       "value": auroc},
                {"baseline": baseline, "split": regime, "metric": "auprc",       "value": auprc},
                {"baseline": baseline, "split": regime, "metric": "f1_tuned",    "value": f1},
                {"baseline": baseline, "split": regime, "metric": "precision",   "value": p},
                {"baseline": baseline, "split": regime, "metric": "recall",      "value": r},
                {"baseline": baseline, "split": regime, "metric": "threshold",   "value": thr},
                {"baseline": baseline, "split": regime, "metric": "hits@1",      "value": h1},
                {"baseline": baseline, "split": regime, "metric": "hits@1_lo",   "value": h1_lo},
                {"baseline": baseline, "split": regime, "metric": "hits@1_hi",   "value": h1_hi},
                {"baseline": baseline, "split": regime, "metric": "hits@3",      "value": h3},
                {"baseline": baseline, "split": regime, "metric": "mrr",         "value": mrr},
                {"baseline": baseline, "split": regime, "metric": "n_anchors",   "value": n_anchors},
                {"baseline": baseline, "split": regime, "metric": "n_test",      "value": int(len(test))},
                {"baseline": baseline, "split": regime, "metric": "n_pos_test",  "value": int(y_test.sum())},
            ])

        # Per-negative-class breakdown (on entire dataset, label=0 only)
        negs = df[df["label"] == 0]
        grp = negs.groupby("source")["score"].agg(["mean", "count"]).reset_index()
        for _, row in grp.iterrows():
            neg_rows.append({
                "baseline": baseline,
                "neg_class": row["source"],
                "mean_score": float(row["mean"]),
                "n": int(row["count"]),
            })

    metrics_df = pd.DataFrame(metric_rows)
    neg_df = pd.DataFrame(neg_rows)
    metrics_path = os.path.join(args.data, "eval_metrics.csv")
    neg_path = os.path.join(args.data, "eval_neg_breakdown.csv")
    metrics_df.to_csv(metrics_path, index=False)
    neg_df.to_csv(neg_path, index=False)

    # Print pivoted summary
    print()
    print("=" * 78)
    print("BASELINE PERFORMANCE  (test-split metrics; F1@train-tuned threshold)")
    print("=" * 78)
    for regime in SPLIT_REGIMES:
        sub = metrics_df[(metrics_df["split"] == regime)
                         & (metrics_df["metric"].isin(
                             ["auroc","auprc","f1_tuned","hits@1","hits@1_lo","hits@1_hi",
                              "hits@3","mrr","n_anchors","n_test","n_pos_test"]))]
        if sub.empty:
            continue
        pv = sub.pivot(index="baseline", columns="metric", values="value")
        order = ["auroc","auprc","f1_tuned","hits@1","hits@1_lo","hits@1_hi",
                 "hits@3","mrr","n_anchors","n_pos_test","n_test"]
        pv = pv[[c for c in order if c in pv.columns]]
        print(f"\n{regime}:")
        print(pv.round(4).to_string())

    print()
    print("=" * 78)
    print("NEGATIVE-CLASS BREAKDOWN  (mean score on label=0 rows; higher = more fooled)")
    print("=" * 78)
    if not neg_df.empty:
        pv = neg_df.pivot(index="baseline", columns="neg_class", values="mean_score")
        print(pv.round(3).to_string())

    print()
    print(f"wrote -> {metrics_path}")
    print(f"wrote -> {neg_path}")

if __name__ == "__main__":
    main()
