"""Enrich and classify successor links.

Reads:
  data/successor_links_raw.parquet  (pred_lei, succ_lei)
  data/entities.parquet              (LEI + all attributes)
  data/events.parquet                (LEI + per-event records)

Writes:
  data/successor_links.parquet
    pred_lei, succ_lei,
    pred_name, succ_name,
    pred_country, succ_country,
    pred_form, succ_form,
    pred_status, succ_status,
    has_real_event, n_real_events,
    classification

Classification uses the same rules as validate_successor_pairs.py so the
numbers reconcile.
"""
import argparse
import os

import pyarrow.parquet as pq
import pyarrow.compute as pc

REAL_EVENT_TYPES = {
    "MERGERS_AND_ACQUISITIONS", "ABSORPTION", "CHANGE_LEGAL_NAME",
    "DEMERGER", "SPINOFF", "BREAKUP", "CHANGE_LEGAL_FORM",
}
LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "LTDA", "PTY", "BHD", "TBK", "SRL", "SPA", "SPRL",
}

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"']:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)

def classify(row):
    if row["succ_name"] is None:
        return "successor_not_found"
    same_name = normalize_name(row["pred_name"]) == normalize_name(row["succ_name"])
    same_country = (row["pred_country"] or "") == (row["succ_country"] or "")
    same_form = (row["pred_form"] or "") == (row["succ_form"] or "")
    has_real = bool(row["has_real_event"])
    if has_real and not same_name:
        return "strong_real"
    if has_real and same_name:
        return "likely_real_event_only"
    if not same_name:
        return "likely_real_name_changed"
    if same_name and same_country and same_form:
        return "likely_administrative"
    return "ambiguous"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="directory containing parquet files")
    args = ap.parse_args()

    raw = pq.read_table(os.path.join(args.data, "successor_links_raw.parquet")).to_pandas()
    print(f"raw successor pairs: {len(raw):,}")

    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code", "entity_status"],
    ).to_pandas().rename(columns={
        "legal_name": "_name",
        "legal_addr_country": "_country",
        "legal_form_code": "_form",
        "entity_status": "_status",
    })

    # Build a per-LEI "has any real-type COMPLETED event" flag
    ev = pq.read_table(
        os.path.join(args.data, "events.parquet"),
        columns=["lei", "event_type", "event_status"],
    ).to_pandas()
    ev_real = ev[(ev["event_status"] == "COMPLETED")
                 & (ev["event_type"].isin(REAL_EVENT_TYPES))]
    real_counts = ev_real.groupby("lei").size().rename("n_real_events").to_frame()
    real_counts["has_real_event"] = True

    # Join: predecessor side
    pred_join = raw.merge(ent, left_on="pred_lei", right_on="lei", how="left").drop(columns=["lei"])
    pred_join = pred_join.rename(columns={
        "_name": "pred_name", "_country": "pred_country",
        "_form": "pred_form", "_status": "pred_status",
    })
    # Join: successor side
    full = pred_join.merge(ent, left_on="succ_lei", right_on="lei", how="left").drop(columns=["lei"])
    full = full.rename(columns={
        "_name": "succ_name", "_country": "succ_country",
        "_form": "succ_form", "_status": "succ_status",
    })
    # Join in event flag on predecessor
    full = full.merge(real_counts, left_on="pred_lei", right_index=True, how="left")
    full["has_real_event"] = full["has_real_event"].fillna(False).astype(bool)
    full["n_real_events"] = full["n_real_events"].fillna(0).astype(int)

    # Classify
    full["classification"] = full.apply(classify, axis=1)

    # Summary
    print()
    print(f"resolved successor records: {full['succ_name'].notna().sum():,}")
    print()
    print("Classification breakdown:")
    counts = full["classification"].value_counts()
    for k, c in counts.items():
        print(f"  {k:32s} {c:>8,}  {100*c/len(full):>5.2f}%")
    plausibly_real = counts.drop(
        ["likely_administrative", "successor_not_found"], errors="ignore").sum()
    print()
    print(f"plausibly-real pairs: {plausibly_real:,}  ({100*plausibly_real/len(full):.2f}%)")

    out_path = os.path.join(args.data, "successor_links.parquet")
    full.to_parquet(out_path, compression="snappy", index=False)
    print()
    print(f"wrote -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
