"""Paired significance test on per-anchor Hits@1 differences.

For each (model_A, model_B, split) triple where both models scored the
same anchor set, computes:
  - McNemar's exact test (textbook for paired binary outcomes)
  - Paired permutation test (no distributional assumption)
  - 95% CI on the mean Hits@1 difference (paired bootstrap)
  - Effect size (mean difference) and discordant pair counts

This is the headline test for "the cross-encoder beats Haiku by 30+pp"
type claims. McNemar is appropriate because each anchor either does or
doesn't get its truth ranked first — paired binary outcome per anchor.

Usage:
  python3 src/eval/paired_significance.py \\
      --pairs pairs.parquet \\
      --model-a method_v2_crossenc --model-b llm_haiku_panel
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def per_anchor_hits(df):
    """Yield (anchor_lei, hit_at_1) for eligible anchors with at least
    one positive and at least 2 candidates."""
    for anchor, sub in df.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        yield anchor, (1 if int(pi[0]) == 0 else 0)


def mcnemar_exact(b, c):
    """Exact McNemar's test (binomial). b = a-wins, c = b-wins (discordant
    pairs only). Returns two-sided p-value."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided: P(X <= k) + P(X >= n-k) under Binomial(n, 0.5)
    p_one_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p_one_tail)


def permutation_test(d, n_perms=10000, seed=42):
    """Paired permutation test on d[i] = h_a[i] - h_b[i] ∈ {-1, 0, 1}.
    Under H0 the sign of each d is random. Returns two-sided p-value."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    n = len(d)
    if n == 0:
        return 1.0
    perms = np.empty(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=n)
        perms[i] = float(np.mean(d * signs))
    return float(np.mean(np.abs(perms) >= abs(obs)))


def paired_bootstrap_ci(d, n_boots=5000, seed=42):
    """95% CI on the mean of d via paired bootstrap."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(d, dtype=float)
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
    ap.add_argument("--pairs", default="pairs.parquet")
    ap.add_argument("--model-a", required=True,
                    help="prediction file stem (e.g. method_v2_crossenc)")
    ap.add_argument("--model-b", required=True,
                    help="prediction file stem to compare against (e.g. llm_haiku_panel)")
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, args.pairs),
        columns=["anchor_lei", "candidate_lei", "label",
                  "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()

    def load(stem):
        p = os.path.join(args.data, "predictions", f"{stem}.parquet")
        return pq.read_table(p).to_pandas()

    pa = load(args.model_a)
    pb = load(args.model_b)
    a_anchors = set(pa["anchor_lei"])
    b_anchors = set(pb["anchor_lei"])
    shared = a_anchors & b_anchors
    print(f"\n{args.model_a}: {len(a_anchors)} anchors", file=sys.stderr)
    print(f"{args.model_b}: {len(b_anchors)} anchors", file=sys.stderr)
    print(f"shared: {len(shared)}", file=sys.stderr)

    splits = ("split_time", "split_event", "split_jurisdiction")
    short = {"split_time": "TIME", "split_event": "EVENT",
              "split_jurisdiction": "JURIS"}

    print()
    print(f"PAIRED SIGNIFICANCE: {args.model_a}  vs  {args.model_b}")
    print(f"{'split':<6s}  {'n':>4s}  "
          f"{'A H@1':>6s}  {'B H@1':>6s}  {'Δ':>6s}  "
          f"{'CI lo':>6s}  {'CI hi':>6s}  "
          f"{'A-only':>6s}  {'B-only':>6s}  "
          f"{'McNemar p':>10s}  {'Perm p':>8s}")
    print("-" * 110)

    for split in splits:
        test = pairs[(pairs[split] == "test") & pairs["anchor_lei"].isin(shared)]
        # Merge in scores for both models on the same test pairs
        da = test.merge(pa[["anchor_lei", "candidate_lei", "score"]],
                          on=["anchor_lei", "candidate_lei"], how="left")
        db = test.merge(pb[["anchor_lei", "candidate_lei", "score"]],
                          on=["anchor_lei", "candidate_lei"], how="left")
        da["score"] = da["score"].fillna(0.0)
        db["score"] = db["score"].fillna(0.0)

        # Per-anchor hits on each model, restricted to anchors that BOTH
        # models scored at least one pair for
        h_a = dict(per_anchor_hits(da))
        h_b = dict(per_anchor_hits(db))
        common = sorted(set(h_a) & set(h_b))
        if not common:
            continue
        ha = np.array([h_a[k] for k in common])
        hb = np.array([h_b[k] for k in common])
        d = ha - hb  # ∈ {-1, 0, 1}

        a_only = int(((ha == 1) & (hb == 0)).sum())
        b_only = int(((ha == 0) & (hb == 1)).sum())
        # both_hit = int(((ha == 1) & (hb == 1)).sum())
        # both_miss = int(((ha == 0) & (hb == 0)).sum())

        ci_lo, ci_hi = paired_bootstrap_ci(d)
        p_mcnemar = mcnemar_exact(a_only, b_only)
        p_perm = permutation_test(d)

        # Format p-values cleanly
        def fmt_p(p):
            if p < 1e-4:
                return "<1e-4"
            if p < 1e-3:
                return f"{p:.4f}"
            return f"{p:.3f}"

        print(f"{short[split]:<6s}  {len(common):>4d}  "
              f"{ha.mean():>6.4f}  {hb.mean():>6.4f}  {d.mean():>+6.3f}  "
              f"{ci_lo:>+6.3f}  {ci_hi:>+6.3f}  "
              f"{a_only:>6d}  {b_only:>6d}  "
              f"{fmt_p(p_mcnemar):>10s}  {fmt_p(p_perm):>8s}")
    print()


if __name__ == "__main__":
    main()
