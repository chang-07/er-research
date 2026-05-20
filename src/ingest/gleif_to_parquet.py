"""Parse GLEIF Level-1 LEI-CDF XML into typed parquet tables.

Outputs into <out_dir>:
  - entities.parquet            (one row per LEI, ~3.3M rows)
  - events.parquet              (one row per LegalEntityEvent, ~1.2M rows)
  - successor_links_raw.parquet (pred_lei -> succ_lei pairs, ~34K rows)

Successor enrichment + classification is done by a separate post-processing
step (build_successor_links.py) that joins successor_links_raw against the
entities table. This keeps the XML parser single-pass.

Level-2 RR (relationships) is handled by a separate ingester once that file
is downloaded.

Usage:
  python -m src.ingest.gleif_to_parquet \\
      --xml 20260519-gleif-concatenated-file-lei2.xml \\
      --out data/

For a smoke test, add --limit 10000 to process only the first N records.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

NS = "http://www.gleif.org/data/schema/leidata/2016"

# ---- schemas --------------------------------------------------------------

ENTITIES_SCHEMA = pa.schema([
    ("lei", pa.string()),
    ("legal_name", pa.string()),
    ("other_names", pa.list_(pa.string())),
    ("legal_form_code", pa.string()),
    ("legal_jurisdiction", pa.string()),
    ("category", pa.string()),
    ("sub_category", pa.string()),
    ("entity_status", pa.string()),
    ("entity_creation_date", pa.timestamp("ms")),
    ("entity_expiration_date", pa.timestamp("ms")),
    ("entity_expiration_reason", pa.string()),
    ("legal_addr_country", pa.string()),
    ("legal_addr_city", pa.string()),
    ("legal_addr_region", pa.string()),
    ("legal_addr_postal_code", pa.string()),
    ("hq_addr_country", pa.string()),
    ("hq_addr_city", pa.string()),
    ("hq_addr_region", pa.string()),
    ("hq_addr_postal_code", pa.string()),
    ("associated_entity_lei", pa.string()),
    ("initial_registration_date", pa.timestamp("ms")),
    ("last_update_date", pa.timestamp("ms")),
    ("next_renewal_date", pa.timestamp("ms")),
    ("registration_status", pa.string()),
    ("validation_sources", pa.string()),
    ("managing_lou", pa.string()),
    ("has_successor", pa.bool_()),
    ("successor_lei", pa.string()),
])

EVENTS_SCHEMA = pa.schema([
    ("lei", pa.string()),
    ("event_type", pa.string()),
    ("event_status", pa.string()),
    ("group_type", pa.string()),
    ("effective_date", pa.timestamp("ms")),
    ("recorded_date", pa.timestamp("ms")),
    ("validation_documents", pa.string()),
    ("affected_field_xpaths", pa.string()),    # ";"-joined
    ("affected_field_values", pa.string()),    # ";"-joined
])

SUCCESSOR_RAW_SCHEMA = pa.schema([
    ("pred_lei", pa.string()),
    ("succ_lei", pa.string()),
])

# ---- helpers --------------------------------------------------------------

def tag(name: str) -> str:
    return f"{{{NS}}}{name}"

def text(elem, name: str) -> Optional[str]:
    """findtext that returns None for missing or empty strings."""
    if elem is None:
        return None
    val = elem.findtext(tag(name))
    if val is None:
        return None
    val = val.strip()
    return val or None

def parse_dt(s: Optional[str]):
    """Parse GLEIF ISO 8601 timestamp; return None for missing/invalid."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        # Python 3.9 fromisoformat handles "+00:00" but not "Z"
        return datetime.fromisoformat(s)
    except ValueError:
        return None

def clear(elem):
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

# ---- per-record extraction ------------------------------------------------

def extract_entity_row(rec) -> dict:
    """Build one entities-table row from an <LEIRecord> element."""
    entity = rec.find(tag("Entity"))
    if entity is None:
        return {"lei": text(rec, "LEI")}  # degenerate; should be rare

    legal_addr = entity.find(tag("LegalAddress"))
    hq_addr = entity.find(tag("HeadquartersAddress"))
    legal_form = entity.find(tag("LegalForm"))
    successor = entity.find(tag("SuccessorEntity"))

    # other names: collect all OtherEntityName under OtherEntityNames
    other_names = []
    oens = entity.find(tag("OtherEntityNames"))
    if oens is not None:
        for n in oens.findall(tag("OtherEntityName")):
            if n.text and n.text.strip():
                other_names.append(n.text.strip())

    # Registration block
    registration = rec.find(tag("Registration"))

    return {
        "lei": text(rec, "LEI"),
        "legal_name": text(entity, "LegalName"),
        "other_names": other_names,
        "legal_form_code": text(legal_form, "EntityLegalFormCode") if legal_form is not None else None,
        "legal_jurisdiction": text(entity, "LegalJurisdiction"),
        "category": text(entity, "EntityCategory"),
        "sub_category": text(entity, "EntitySubCategory"),
        "entity_status": text(entity, "EntityStatus"),
        "entity_creation_date": parse_dt(text(entity, "EntityCreationDate")),
        "entity_expiration_date": parse_dt(text(entity, "EntityExpirationDate")),
        "entity_expiration_reason": text(entity, "EntityExpirationReason"),
        "legal_addr_country": text(legal_addr, "Country"),
        "legal_addr_city": text(legal_addr, "City"),
        "legal_addr_region": text(legal_addr, "Region"),
        "legal_addr_postal_code": text(legal_addr, "PostalCode"),
        "hq_addr_country": text(hq_addr, "Country"),
        "hq_addr_city": text(hq_addr, "City"),
        "hq_addr_region": text(hq_addr, "Region"),
        "hq_addr_postal_code": text(hq_addr, "PostalCode"),
        "associated_entity_lei": text(entity.find(tag("AssociatedEntity")) if entity.find(tag("AssociatedEntity")) is not None else entity, "AssociatedLEI"),
        "initial_registration_date": parse_dt(text(registration, "InitialRegistrationDate")),
        "last_update_date": parse_dt(text(registration, "LastUpdateDate")),
        "next_renewal_date": parse_dt(text(registration, "NextRenewalDate")),
        "registration_status": text(registration, "RegistrationStatus"),
        "validation_sources": text(registration, "ValidationSources"),
        "managing_lou": text(registration, "ManagingLOU"),
        "has_successor": successor is not None,
        "successor_lei": text(successor, "SuccessorLEI") if successor is not None else None,
    }

def extract_event_rows(rec, lei: str) -> list:
    """Build zero-or-more events-table rows from a record."""
    entity = rec.find(tag("Entity"))
    if entity is None:
        return []
    events = entity.find(tag("LegalEntityEvents"))
    if events is None:
        return []
    out = []
    for ev in events.findall(tag("LegalEntityEvent")):
        affected = ev.find(tag("AffectedFields"))
        xpaths, values = [], []
        if affected is not None:
            for af in affected.findall(tag("AffectedField")):
                xpaths.append(af.get("field_xpath") or "")
                values.append((af.text or "").strip())
        out.append({
            "lei": lei,
            "event_type": text(ev, "LegalEntityEventType"),
            "event_status": ev.get("event_status"),
            "group_type": ev.get("group_type"),
            "effective_date": parse_dt(text(ev, "LegalEntityEventEffectiveDate")),
            "recorded_date": parse_dt(text(ev, "LegalEntityEventRecordedDate")),
            "validation_documents": text(ev, "ValidationDocuments"),
            "affected_field_xpaths": "; ".join(xpaths) if xpaths else None,
            "affected_field_values": "; ".join(values) if values else None,
        })
    return out

# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="path to GLEIF Level-1 concatenated XML")
    ap.add_argument("--out", required=True, help="output directory for parquet files")
    ap.add_argument("--batch-size", type=int, default=50_000,
                    help="rows per parquet row-group flush")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many LEIRecords (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ent_path = os.path.join(args.out, "entities.parquet")
    ev_path = os.path.join(args.out, "events.parquet")
    succ_path = os.path.join(args.out, "successor_links_raw.parquet")

    ent_writer = pq.ParquetWriter(ent_path, ENTITIES_SCHEMA, compression="snappy")
    ev_writer = pq.ParquetWriter(ev_path, EVENTS_SCHEMA, compression="snappy")
    succ_writer = pq.ParquetWriter(succ_path, SUCCESSOR_RAW_SCHEMA, compression="snappy")

    ent_buf, ev_buf, succ_buf = [], [], []
    total = 0
    total_events = 0
    total_succ = 0
    t0 = time.time()
    last_log = t0

    def flush(buf, writer, schema):
        if not buf:
            return
        tbl = pa.Table.from_pylist(buf, schema=schema)
        writer.write_table(tbl)
        buf.clear()

    try:
        ctx = etree.iterparse(args.xml, tag=tag("LEIRecord"), events=("end",))
        for _, rec in ctx:
            row = extract_entity_row(rec)
            lei = row["lei"]
            ent_buf.append(row)
            if row.get("has_successor") and row.get("successor_lei"):
                succ_buf.append({"pred_lei": lei, "succ_lei": row["successor_lei"]})
                total_succ += 1
            evs = extract_event_rows(rec, lei)
            if evs:
                ev_buf.extend(evs)
                total_events += len(evs)

            total += 1
            if len(ent_buf) >= args.batch_size:
                flush(ent_buf, ent_writer, ENTITIES_SCHEMA)
            if len(ev_buf) >= args.batch_size:
                flush(ev_buf, ev_writer, EVENTS_SCHEMA)
            if len(succ_buf) >= args.batch_size:
                flush(succ_buf, succ_writer, SUCCESSOR_RAW_SCHEMA)
            clear(rec)

            now = time.time()
            if now - last_log > 15:
                elapsed = now - t0
                rate = total / elapsed
                print(f"  [{elapsed:6.1f}s] {total:>10,} records "
                      f"({rate:>7,.0f}/s) | events={total_events:,} succ={total_succ:,}",
                      flush=True)
                last_log = now

            if args.limit is not None and total >= args.limit:
                break

        flush(ent_buf, ent_writer, ENTITIES_SCHEMA)
        flush(ev_buf, ev_writer, EVENTS_SCHEMA)
        flush(succ_buf, succ_writer, SUCCESSOR_RAW_SCHEMA)
    finally:
        ent_writer.close()
        ev_writer.close()
        succ_writer.close()

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"DONE in {elapsed:.1f}s ({total/elapsed:,.0f} rec/s)")
    print(f"  entities                 : {total:>10,}  -> {ent_path}")
    print(f"  events                   : {total_events:>10,}  -> {ev_path}")
    print(f"  successor_links_raw      : {total_succ:>10,}  -> {succ_path}")
    print()
    print(f"  entities.parquet         : {os.path.getsize(ent_path)/1e6:>8.1f} MB")
    print(f"  events.parquet           : {os.path.getsize(ev_path)/1e6:>8.1f} MB")
    print(f"  successor_links_raw      : {os.path.getsize(succ_path)/1e6:>8.1f} MB")

if __name__ == "__main__":
    main()
