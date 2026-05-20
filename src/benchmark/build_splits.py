"""Assemble the final TemporalLink v1 pair dataset with three eval splits.

For each plausibly-real successor pair, finds the most informative COMPLETED
lifecycle event on the predecessor (the "anchor event") and uses its date,
type, and the predecessor's jurisdiction to assign the pair to one of three
split regimes:

  split_time
    train: anchor event before 2025-01-01
    test:  anchor event on/after 2025-01-01
    excl:  no anchor event date

  split_event
    train: anchor event in {MERGERS_AND_ACQUISITIONS, CHANGE_LEGAL_NAME,
                             CHANGE_LEGAL_FORM}
    test:  anchor event in {SPINOFF, DEMERGER, BREAKUP, ABSORPTION}
    excl:  no anchor event, or event type outside both sets

  split_jurisdiction
    train: predecessor legal_addr_country in {US, GB, DE}
    test:  predecessor legal_addr_country in {SG, JP, BR}
    excl:  other countries (most of the dataset)

For each split, each positive is paired with up to K hard negatives (default
5 in train, 10 in test). Train and test negatives are deduplicated per
positive and never reused across splits' label=0 cells.

Output: data/pairs.parquet
  anchor_lei, candidate_lei, label (1 = same economic entity), source,
  anchor_event_date, anchor_event_type, anchor_country,
  split_time, split_event, split_jurisdiction
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REAL_EVENT_TYPES = [
    "MERGERS_AND_ACQUISITIONS", "ABSORPTION", "CHANGE_LEGAL_NAME",
    "DEMERGER", "SPINOFF", "BREAKUP", "CHANGE_LEGAL_FORM",
]
# event-type split definitions
EVENT_TRAIN = {"MERGERS_AND_ACQUISITIONS", "CHANGE_LEGAL_NAME", "CHANGE_LEGAL_FORM"}
EVENT_TEST = {"SPINOFF", "DEMERGER", "BREAKUP", "ABSORPTION"}

# event-type priority for picking the *anchor* event when a predecessor has many
EVENT_PRIORITY = {
    "MERGERS_AND_ACQUISITIONS": 10,
    "ABSORPTION": 9,
    "SPINOFF": 8,
    "DEMERGER": 8,
    "BREAKUP": 7,
    "CHANGE_LEGAL_NAME": 6,
    "CHANGE_LEGAL_FORM": 5,
}

TIME_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
JURIS_TRAIN = {"US", "GB", "DE"}
JURIS_TEST = {"SG", "JP", "BR"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--negs-train", type=int, default=5)
    ap.add_argument("--negs-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading positives...", file=sys.stderr)
    pos = pq.read_table(
        os.path.join(args.data, "successor_links.parquet"),
        columns=["pred_lei", "succ_lei", "classification", "pred_country"],
    ).to_pandas()
    pos = pos[pos["classification"] != "likely_administrative"].copy()
    print(f"  {len(pos):,} plausibly-real positives", file=sys.stderr)

    print("Loading events...", file=sys.stderr)
    ev = pq.read_table(
        os.path.join(args.data, "events.parquet"),
        columns=["lei", "event_type", "event_status", "effective_date"],
    ).to_pandas()
    ev = ev[(ev["event_status"] == "COMPLETED")
            & (ev["event_type"].isin(REAL_EVENT_TYPES))]
    ev["priority"] = ev["event_type"].map(EVENT_PRIORITY).fillna(0)
    # pick highest-priority event per LEI; tie-break by latest date
    ev = ev.sort_values(["lei", "priority", "effective_date"],
                        ascending=[True, False, False])
    anchor_ev = ev.groupby("lei").first().reset_index()
    anchor_ev = anchor_ev.rename(columns={
        "event_type": "anchor_event_type",
        "effective_date": "anchor_event_date",
    })[["lei", "anchor_event_type", "anchor_event_date"]]
    print(f"  picked anchor events for {len(anchor_ev):,} LEIs", file=sys.stderr)

    pos = pos.merge(anchor_ev, left_on="pred_lei", right_on="lei", how="left").drop(columns=["lei"])
    pos = pos.rename(columns={"pred_country": "anchor_country"})

    # Normalize the anchor_event_date to UTC tz-aware for clean comparisons
    pos["anchor_event_date"] = pd.to_datetime(pos["anchor_event_date"], utc=True, errors="coerce")

    # Split assignments
    def time_split(d):
        if pd.isna(d):
            return "excl"
        return "train" if d < TIME_CUTOFF else "test"

    def event_split(t):
        if t in EVENT_TRAIN: return "train"
        if t in EVENT_TEST:  return "test"
        return "excl"

    def juris_split(c):
        if c in JURIS_TRAIN: return "train"
        if c in JURIS_TEST:  return "test"
        return "excl"

    pos["split_time"] = pos["anchor_event_date"].apply(time_split)
    pos["split_event"] = pos["anchor_event_type"].apply(event_split)
    pos["split_jurisdiction"] = pos["anchor_country"].apply(juris_split)

    print()
    print("Split coverage on positives:")
    for col in ("split_time", "split_event", "split_jurisdiction"):
        vc = pos[col].value_counts()
        print(f"  {col:22s} train={vc.get('train',0):>6,}  "
              f"test={vc.get('test',0):>5,}  excl={vc.get('excl',0):>6,}")
    print()

    # Load negatives
    print("Loading hard negatives...", file=sys.stderr)
    negs = pq.read_table(os.path.join(args.data, "hard_negatives.parquet")).to_pandas()
    negs = negs.rename(columns={"pred_lei": "anchor_lei", "neg_lei": "candidate_lei",
                                 "negative_class": "source"})
    # Shuffle for fair sampling per positive
    negs = negs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Group negatives per anchor (predecessor) LEI
    neg_groups = negs.groupby("anchor_lei").indices  # dict: lei -> array of row indices

    # Build pair rows
    print("Assembling pair rows...", file=sys.stderr)
    pair_rows = []

    for _, row in pos.iterrows():
        pred = row["pred_lei"]
        succ = row["succ_lei"]
        e_date = row["anchor_event_date"]
        e_type = row["anchor_event_type"]
        e_country = row["anchor_country"]
        st = row["split_time"]
        se = row["split_event"]
        sj = row["split_jurisdiction"]

        # Positive
        pair_rows.append({
            "anchor_lei": pred,
            "candidate_lei": succ,
            "label": 1,
            "source": "successor_link",
            "anchor_event_date": e_date,
            "anchor_event_type": e_type,
            "anchor_country": e_country,
            "split_time": st,
            "split_event": se,
            "split_jurisdiction": sj,
        })

        # Determine how many negatives this anchor gets based on its splits.
        # Use the max(K) requested across the splits the anchor participates in,
        # so train and test sets always have enough.
        any_test = (st == "test") or (se == "test") or (sj == "test")
        k = args.negs_test if any_test else args.negs_train

        idx = neg_groups.get(pred)
        if idx is None:
            continue
        chosen = list(idx[:k])
        for i in chosen:
            n = negs.iloc[i]
            pair_rows.append({
                "anchor_lei": pred,
                "candidate_lei": n["candidate_lei"],
                "label": 0,
                "source": n["source"],
                "anchor_event_date": e_date,
                "anchor_event_type": e_type,
                "anchor_country": e_country,
                "split_time": st,
                "split_event": se,
                "split_jurisdiction": sj,
            })

    df = pd.DataFrame(pair_rows)
    print()
    print(f"Total pair rows: {len(df):,}")
    print(f"  positives (label=1): {(df['label']==1).sum():,}")
    print(f"  negatives (label=0): {(df['label']==0).sum():,}")
    print()
    print("Split breakdown (rows per split x train/test/excl):")
    for col in ("split_time", "split_event", "split_jurisdiction"):
        vc = df.groupby([col, "label"]).size().unstack(fill_value=0)
        print(f"\n  {col}:")
        print(vc.to_string())
    print()

    out = os.path.join(args.data, "pairs.parquet")
    df.to_parquet(out, compression="snappy", index=False)
    print(f"wrote -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
