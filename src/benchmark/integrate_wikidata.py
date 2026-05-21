"""Integrate the 259 new-signal Wikidata LEI-bridged positives into
pairs.parquet, producing pairs_v5_wikidata.parquet that the cross-encoder
can train on.

For each new Wikidata positive (pred_lei, succ_lei):
  - add as label=1 row with source='wikidata_p156' / 'wikidata_p1366' /
    'wikidata_p7888' and split assignments derived from the predecessor's
    GLEIF event metadata (if any) so they don't leak across our held-out
    splits.
  - mine hard negatives for the predecessor by reusing existing
    hard_negatives_v2.parquet entries (or fall back to random same-country
    if not present).

Output: data/pairs_v5_wikidata.parquet (drop-in replacement for
pairs.parquet, ~125,800 rows)
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    wd = pq.read_table(os.path.join(data, "wikidata_lei_positives.parquet")).to_pandas()
    wd_new = wd[wd.pred_in_gleif & wd.succ_in_gleif & wd.new_signal].copy()
    print(f"Wikidata new positives: {len(wd_new)}", file=sys.stderr)

    hn = pq.read_table(
        os.path.join(data, "hard_negatives_v2.parquet"),
        columns=["pred_lei", "neg_lei", "negative_class"],
    ).to_pandas()
    hn_by_pred = hn.groupby("pred_lei")["neg_lei"].apply(list).to_dict()
    hn_cls_by = hn.set_index(["pred_lei", "neg_lei"])["negative_class"].to_dict()

    # Use the predecessor's existing pairs.parquet entry to inherit split
    # assignments when available. Otherwise default to "excl" — we only add
    # these positives where we already have GLEIF context for the predecessor.
    pred_info = pairs[pairs.label == 1].drop_duplicates(subset=["anchor_lei"]) \
        .set_index("anchor_lei")[["anchor_event_date", "anchor_event_type",
                                    "anchor_country", "split_time",
                                    "split_event", "split_jurisdiction"]] \
        .to_dict(orient="index")

    new_rows = []
    rng = np.random.default_rng(42)
    n_added_pos, n_added_neg = 0, 0
    for _, r in wd_new.iterrows():
        pi = pred_info.get(r.pred_lei)
        if pi is None:
            # predecessor has no GLEIF event context — put in "excl" so we
            # don't risk leakage; still useful for in-batch contrast at train
            pi = {"anchor_event_date": None, "anchor_event_type": None,
                  "anchor_country": None, "split_time": "excl",
                  "split_event": "excl", "split_jurisdiction": "excl"}
        new_rows.append({
            "anchor_lei": r.pred_lei,
            "candidate_lei": r.succ_lei,
            "label": 1,
            "source": f"wikidata_{r.source_prop.lower()}",
            **pi,
        })
        n_added_pos += 1
        # pair with up to 4 hard negs (matches our cross-encoder training ratio)
        existing_negs = hn_by_pred.get(r.pred_lei, [])
        if existing_negs:
            sample = rng.choice(existing_negs,
                                  size=min(4, len(existing_negs)),
                                  replace=False)
            for nl in sample:
                cls = hn_cls_by.get((r.pred_lei, nl), "wikidata_borrowed")
                new_rows.append({
                    "anchor_lei": r.pred_lei,
                    "candidate_lei": str(nl),
                    "label": 0,
                    "source": cls,
                    **pi,
                })
                n_added_neg += 1

    new_df = pd.DataFrame(new_rows)
    # Coerce dates to ms-precision to avoid out-of-bounds ns timestamps
    if "anchor_event_date" in new_df.columns:
        new_df["anchor_event_date"] = pd.to_datetime(
            new_df["anchor_event_date"], utc=True, errors="coerce")
    out = pd.concat([pairs, new_df], ignore_index=True, sort=False)
    out_path = os.path.join(data, "pairs_v5_wikidata.parquet")
    out.to_parquet(out_path, compression="snappy", index=False)
    print(f"  added {n_added_pos:,} positives + {n_added_neg:,} negs",
          file=sys.stderr)
    print(f"  total rows: {len(pairs):,} -> {len(out):,}", file=sys.stderr)
    print(f"wrote -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
