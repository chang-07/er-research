"""Parse GLEIF Level-2 Relationship Record (RR) XML into a parquet table.

Outputs:
  data/relationships.parquet
    start_lei, end_lei,
    relationship_type,           # e.g. IS_DIRECTLY_CONSOLIDATED_BY
    relationship_status,         # ACTIVE / INACTIVE
    start_date, end_date,        # relationship period
    registration_status,         # of the RR record itself (ISSUED / LAPSED / ...)
    managing_lou

Convention: start_node is the child / dependent entity; end_node is the
parent / managing entity. For example, an IS_DIRECTLY_CONSOLIDATED_BY edge
means start_lei is consolidated by end_lei.
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

NS = "http://www.gleif.org/data/schema/rr/2016"

RR_SCHEMA = pa.schema([
    ("start_lei", pa.string()),
    ("end_lei", pa.string()),
    ("relationship_type", pa.string()),
    ("relationship_status", pa.string()),
    ("start_date", pa.timestamp("ms")),
    ("end_date", pa.timestamp("ms")),
    ("registration_status", pa.string()),
    ("managing_lou", pa.string()),
])

def tag(name: str) -> str:
    return f"{{{NS}}}{name}"

def text(elem, name: str) -> Optional[str]:
    if elem is None:
        return None
    val = elem.findtext(tag(name))
    if val is None:
        return None
    val = val.strip()
    return val or None

def parse_dt(s: Optional[str]):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None

def clear(elem):
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

def extract_row(rec):
    relationship = rec.find(tag("Relationship"))
    registration = rec.find(tag("Registration"))
    if relationship is None:
        return None
    start_node = relationship.find(tag("StartNode"))
    end_node = relationship.find(tag("EndNode"))
    if start_node is None or end_node is None:
        return None

    # Pick the first (most common) RelationshipPeriod with type=RELATIONSHIP_PERIOD
    start_date = None
    end_date = None
    periods = relationship.find(tag("RelationshipPeriods"))
    if periods is not None:
        for p in periods.findall(tag("RelationshipPeriod")):
            ptype = text(p, "PeriodType")
            if ptype == "RELATIONSHIP_PERIOD":
                start_date = parse_dt(text(p, "StartDate"))
                end_date = parse_dt(text(p, "EndDate"))
                break

    return {
        "start_lei": text(start_node, "NodeID"),
        "end_lei": text(end_node, "NodeID"),
        "relationship_type": text(relationship, "RelationshipType"),
        "relationship_status": text(relationship, "RelationshipStatus"),
        "start_date": start_date,
        "end_date": end_date,
        "registration_status": text(registration, "RegistrationStatus"),
        "managing_lou": text(registration, "ManagingLOU"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="path to GLEIF RR concatenated XML")
    ap.add_argument("--out", required=True, help="output directory for parquet")
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "relationships.parquet")
    writer = pq.ParquetWriter(out_path, RR_SCHEMA, compression="snappy")

    buf = []
    total = 0
    skipped = 0
    t0 = time.time()
    last_log = t0

    def flush():
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=RR_SCHEMA))
            buf.clear()

    try:
        ctx = etree.iterparse(args.xml, tag=tag("RelationshipRecord"), events=("end",))
        for _, rec in ctx:
            row = extract_row(rec)
            if row is None:
                skipped += 1
            else:
                buf.append(row)
            total += 1
            if len(buf) >= args.batch_size:
                flush()
            clear(rec)

            now = time.time()
            if now - last_log > 15:
                elapsed = now - t0
                print(f"  [{elapsed:6.1f}s] {total:>8,} records ({total/elapsed:,.0f}/s)",
                      flush=True)
                last_log = now
        flush()
    finally:
        writer.close()

    elapsed = time.time() - t0
    print()
    print(f"DONE in {elapsed:.1f}s ({total/elapsed:,.0f} rec/s)")
    print(f"  total records   : {total:>10,}")
    print(f"  skipped (no rel): {skipped:>10,}")
    print(f"  -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
