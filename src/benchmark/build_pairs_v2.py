"""Assemble a pairs-format pool from the v2 (E5-cosine) hard negatives.

The cross-encoder trains on E5-cosine negatives while the released
`features_gbm` baseline trains on the v1 token-overlap pool, so every
off-pool comparison between them confounds architecture with negative
source. This builds the missing cell: the same positives and the same
splits as `pairs.parquet`, with negatives drawn from
`hard_negatives_v2.parquet`, so a GBM can be trained on exactly the
negatives the cross-encoder saw.

Caps match the released pool: up to 5 negatives per positive when the
anchor is train in every regime, up to 10 when any regime holds it out.

Usage: python3 src/data/build_pairs_v2.py --data data
Output: data/pairs_v2.parquet
"""
import argparse
import os

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="pairs_v2.parquet")
    args = ap.parse_args()

    pairs = pq.read_table(os.path.join(args.data, "pairs.parquet")).to_pandas()
    negs = pq.read_table(os.path.join(args.data,
                                      "hard_negatives_v2.parquet")).to_pandas()

    pos = pairs[pairs["label"] == 1].copy()
    print(f"positives carried over: {len(pos):,}")

    # Split labels and anchor metadata travel with the anchor, so a
    # negative inherits whatever its predecessor already had.
    meta_cols = ["anchor_event_date", "anchor_event_type", "anchor_country",
                 "split_time", "split_event", "split_jurisdiction"]
    meta = (pos.drop_duplicates("anchor_lei")
               .set_index("anchor_lei")[meta_cols])

    negs = negs[negs["pred_lei"].isin(meta.index)].copy()
    negs = negs.rename(columns={"pred_lei": "anchor_lei",
                                "neg_lei": "candidate_lei",
                                "negative_class": "source"})

    held = meta[(meta["split_time"] == "test")
                | (meta["split_event"] == "test")
                | (meta["split_jurisdiction"] == "test")].index
    cap = pd.Series(5, index=meta.index)
    cap.loc[cap.index.isin(held)] = 10

    # Highest-scoring (hardest) negatives first, then take each anchor's cap.
    negs = negs.sort_values(["anchor_lei", "score"], ascending=[True, False])
    negs["rank"] = negs.groupby("anchor_lei").cumcount()
    negs = negs[negs["rank"] < negs["anchor_lei"].map(cap)].drop(columns=["rank", "score"])
    negs["label"] = 0
    negs = negs.join(meta, on="anchor_lei")

    out = pd.concat([pos[negs.columns], negs], ignore_index=True)
    out = out.drop_duplicates(["anchor_lei", "candidate_lei"])
    path = os.path.join(args.data, args.out)
    out.to_parquet(path, compression="snappy", index=False)

    print(f"negatives kept: {int((out['label'] == 0).sum()):,}")
    print(f"wrote {path}: {len(out):,} rows, "
          f"{out['anchor_lei'].nunique():,} anchors")
    print(out["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
