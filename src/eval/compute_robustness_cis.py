"""Fast bootstrap CIs for the 2x2 train/eval matrix of cross-encoder cells.

Avoids the per-anchor groupby that made compare_distributions.py slow on
bootstrap mode. Vectorized: for each (train, eval, split) cell, computes
per-anchor rank once, then bootstraps over anchors.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CELLS = [
    # (train_marker, eval_pairs_file, pred_file)
    ("trn-v2", "pairs.parquet",    "method_v2_crossenc_repro"),
    ("trn-v2", "pairs_v3.parquet", "method_v2_crossenc_repro_on_pairs_v3"),
    ("trn-v3", "pairs.parquet",    "method_v3_crossenc_on_pairs"),
    ("trn-v3", "pairs_v3.parquet", "method_v3_crossenc"),
    # baselines (for reference deltas)
    ("E5-zs",  "pairs.parquet",    "sentence_encoder"),
    ("E5-zs",  "pairs_v3.parquet", "sentence_encoder_v3"),
    # ORIGINAL paper predictions (for sanity check)
    ("trn-v2 (paper)", "pairs.parquet", "method_v2_crossenc"),
]

SPLITS = ["split_time", "split_event", "split_jurisdiction"]


def per_anchor_ranks(df_test):
    """Vectorized per-anchor rank-of-first-positive.

    Skips anchors with fewer than 2 candidates (the test set doesn't have a
    hard negative for them, so Hits@1 would trivially be 1). Matches
    evaluate.py's retrieval_metrics logic exactly.
    """
    rng = np.random.default_rng(0)
    d = df_test.copy()
    d["_tb"] = rng.random(len(d))
    d = d.sort_values(["anchor_lei", "score", "_tb"],
                      ascending=[True, False, False])
    d["rank"] = d.groupby("anchor_lei", sort=False).cumcount() + 1
    # candidate count per anchor
    cand_counts = d.groupby("anchor_lei", sort=False).size()
    multi_anchors = cand_counts[cand_counts >= 2].index
    d_multi = d[d["anchor_lei"].isin(multi_anchors)]
    pos = d_multi[d_multi["label"] == 1]
    first_pos = pos.groupby("anchor_lei", sort=False)["rank"].min()
    return first_pos.values


def main():
    data = "data"
    pairs_cache = {}

    def get_pairs(f):
        if f not in pairs_cache:
            pairs_cache[f] = pq.read_table(
                os.path.join(data, f),
                columns=["anchor_lei", "candidate_lei", "label",
                          "split_time", "split_event", "split_jurisdiction"],
            ).to_pandas()
        return pairs_cache[f]

    rows = []
    for train_marker, pairs_file, pred_stem in CELLS:
        ppath = os.path.join(data, "predictions", f"{pred_stem}.parquet")
        if not os.path.exists(ppath):
            print(f"SKIP {pred_stem}: not found", file=sys.stderr)
            continue
        preds = pq.read_table(ppath).to_pandas()
        pairs = get_pairs(pairs_file)
        df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0.0)

        for split in SPLITS:
            test = df[df[split] == "test"]
            if len(test) == 0 or test["label"].nunique() < 2:
                continue
            ranks = per_anchor_ranks(test)
            if len(ranks) == 0:
                continue
            is_hit = (ranks == 1).astype(np.int8)
            h1 = float(is_hit.mean())
            # bootstrap
            rng = np.random.default_rng(42)
            boots = np.empty(500)
            for i in range(500):
                idx = rng.integers(0, len(is_hit), size=len(is_hit))
                boots[i] = is_hit[idx].mean()
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            rows.append({
                "model": train_marker,
                "eval_pool": pairs_file,
                "split": split,
                "hits1": h1,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_half": (hi - lo) / 2,
                "n_anchors": int(len(ranks)),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = os.path.join(data, "robustness_cis.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
