"""Track Hits@1 improvements across the v4 progression.

Reports a side-by-side table of:
  baseline (v2_crossenc) | v4 enriched (s=42) | v4 ensemble (avg 3 seeds)
    | v4 enriched + symmetric scoring

Per (split, model) cell: Hits@1 + bootstrap 95% CI + delta vs baseline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VARIANTS = [
    ("baseline (v2 CE, MiniLM-L-6)", "method_v2_crossenc"),
    ("v4 enriched (seed=42)",         "method_v4_crossenc_enriched"),
    ("v4 enriched (seed=7)",          "method_v4_crossenc_enriched_s7"),
    ("v4 enriched (seed=1337)",       "method_v4_crossenc_enriched_s1337"),
    ("v4 enriched + symmetric",       "method_v4_crossenc_enriched_sym"),
    ("v5 + Wikidata-LEI positives",   "method_v5_crossenc_wikidata"),
    ("v6 + country/juris expansion",  "method_v6_crossenc_v6prompt"),
    ("v7 FINAL: ens3 + v5 + v6 avg",  "method_v7_final_ensemble"),
]


def hits1(d):
    hits, n = 0, 0
    for _, sub in d.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        hits += 1 if int(pi[0]) == 0 else 0
        n += 1
    return (hits / n) if n else 0.0, n


def bootstrap_ci(d, n_boots=200, seed=42):
    """Bootstrap 95% CI on Hits@1 by resampling anchors."""
    anchors, ranks = [], []
    for anchor, sub in d.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        anchors.append(anchor)
        ranks.append(int(pi[0]) + 1)
    if not ranks:
        return float("nan"), float("nan")
    is_hit = np.array([1 if r == 1 else 0 for r in ranks])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boots)
    for i in range(n_boots):
        idx = rng.integers(0, len(is_hit), size=len(is_hit))
        boots[i] = is_hit[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()

    # Compute ensemble file on the fly if 3-seed predictions exist
    seed_paths = [
        os.path.join(data, "predictions", f"method_v4_crossenc_enriched{suf}.parquet")
        for suf in ["", "_s7", "_s1337"]
    ]
    if all(os.path.exists(p) for p in seed_paths):
        print("building 3-seed ensemble...", file=sys.stderr)
        dfs = [pq.read_table(p).to_pandas() for p in seed_paths]
        ens = dfs[0][["anchor_lei", "candidate_lei"]].copy()
        ens["score"] = sum(df["score"] for df in dfs) / 3.0
        ens_path = os.path.join(data, "predictions",
                                  "method_v4_crossenc_enriched_ens3.parquet")
        ens.to_parquet(ens_path, compression="snappy", index=False)
        print(f"  wrote ensemble -> {ens_path}", file=sys.stderr)
        VARIANTS.append(("v4 enriched ensemble (avg 3 seeds)",
                         "method_v4_crossenc_enriched_ens3"))

    print()
    print(f"{'method':45s}  {'split':<25s}  {'Hits@1':>8s}  {'CI lo':>7s}  {'CI hi':>7s}  {'Δ vs v2':>8s}")
    print("-" * 110)

    # First compute baseline Hits@1 for each split
    base_path = os.path.join(data, "predictions", "method_v2_crossenc.parquet")
    base_pred = pq.read_table(base_path).to_pandas()
    base_df = pairs.merge(base_pred, on=["anchor_lei", "candidate_lei"], how="left")
    base_df["score"] = base_df["score"].fillna(0)
    base_h = {}
    for split in ("split_time", "split_event", "split_jurisdiction"):
        h, _ = hits1(base_df[base_df[split] == "test"])
        base_h[split] = h

    for label, stem in VARIANTS:
        ppath = os.path.join(data, "predictions", f"{stem}.parquet")
        if not os.path.exists(ppath):
            print(f"  SKIP {label}: {ppath} not present")
            continue
        preds = pq.read_table(ppath).to_pandas()
        df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0)
        for split in ("split_time", "split_event", "split_jurisdiction"):
            test = df[df[split] == "test"]
            h, n = hits1(test)
            lo, hi = bootstrap_ci(test)
            delta = (h - base_h[split]) * 100
            print(f"{label:45s}  {split:<25s}  {h:>8.4f}  {lo:>7.4f}  {hi:>7.4f}  {delta:>+7.2f}pp")
        print()


if __name__ == "__main__":
    main()
