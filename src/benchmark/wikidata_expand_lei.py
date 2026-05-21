"""Pull additional successor pairs from Wikidata via the LEI bridge.

The existing wikidata_marquee.py uses P156 (followed by) + fuzzy name
matching to GLEIF, yielding 302 both-side-resolved pairs but with name-
match noise. This script does it more cleanly:

  - Query Wikidata for ALL pairs where BOTH sides have a P1278 (LEI)
    AND a successor-like relationship: P156 (followed by), P1366
    (replaced by), or P7888 (merged into).
  - LEI bridge means we don't need name matching — both LEIs are
    explicit, so cross-jurisdiction / language-mismatch cases that
    P156 + name match would lose are recovered.
  - Output: data/wikidata_lei_positives.parquet with
      pred_lei, succ_lei, source_prop ('P156'|'P1366'|'P7888'), wd_pred, wd_succ

Usage:
  python3 src/benchmark/wikidata_expand_lei.py --data data
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

SPARQL_URL = "https://query.wikidata.org/sparql"
UA = "er-research/0.2 admin@chang-07.dev"

SUCCESSOR_PROPS = ["P156", "P1366", "P7888"]
PROP_DESCRIPTIONS = {
    "P156":  "followed by",
    "P1366": "replaced by",
    "P7888": "merged into",
}


def build_query(prop):
    return f"""
SELECT ?pred ?predLabel ?predLei ?succ ?succLabel ?succLei WHERE {{
  ?pred wdt:{prop} ?succ .
  ?pred wdt:P1278 ?predLei .
  ?succ wdt:P1278 ?succLei .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 10000
"""


def run_sparql(query, max_retries=4):
    """Hits the Wikidata SPARQL endpoint. During active WDQS outages the
    endpoint can throttle to 1 req/min; we sleep + retry up to max_retries
    times before giving up.
    """
    qs = urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(
        SPARQL_URL + "?" + qs,
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
    )
    delay = 65
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as f:
                return json.loads(f.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                print(f"  429 — sleeping {delay}s before retry "
                      f"({attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 180)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    rows = []
    for prop in SUCCESSOR_PROPS:
        print(f"querying Wikidata for {prop} ({PROP_DESCRIPTIONS[prop]})...",
              file=sys.stderr)
        try:
            res = run_sparql(build_query(prop))
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        bindings = res.get("results", {}).get("bindings", [])
        print(f"  got {len(bindings)} bindings", file=sys.stderr)
        for b in bindings:
            rows.append({
                "wd_pred":   b["pred"]["value"].rsplit("/", 1)[-1],
                "pred_lei":  b["predLei"]["value"].strip(),
                "wd_pred_label": b.get("predLabel", {}).get("value", ""),
                "wd_succ":   b["succ"]["value"].rsplit("/", 1)[-1],
                "succ_lei":  b["succLei"]["value"].strip(),
                "wd_succ_label": b.get("succLabel", {}).get("value", ""),
                "source_prop": prop,
            })
        time.sleep(70)  # be polite (currently rate-limited)

    df = pd.DataFrame(rows)
    print(f"\nTotal rows: {len(df)}", file=sys.stderr)
    if df.empty:
        print("no data — aborting", file=sys.stderr)
        sys.exit(1)

    # dedupe (a pair might appear under multiple properties)
    df = df.drop_duplicates(subset=["pred_lei", "succ_lei"], keep="first")
    print(f"After dedupe: {len(df)}", file=sys.stderr)

    # Filter to LEIs we have in our entity universe
    ent = pq.read_table(os.path.join(args.data, "entities.parquet"),
                         columns=["lei"]).to_pandas()
    our_leis = set(ent["lei"])
    df["pred_in_gleif"] = df["pred_lei"].isin(our_leis)
    df["succ_in_gleif"] = df["succ_lei"].isin(our_leis)
    print(f"  predLei in our GLEIF: {df.pred_in_gleif.sum()}", file=sys.stderr)
    print(f"  succLei in our GLEIF: {df.succ_in_gleif.sum()}", file=sys.stderr)
    print(f"  BOTH in our GLEIF: {(df.pred_in_gleif & df.succ_in_gleif).sum()}",
          file=sys.stderr)

    # How many are NEW (not already in our successor_links)?
    sl = pq.read_table(os.path.join(args.data, "successor_links.parquet"),
                        columns=["pred_lei", "succ_lei"]).to_pandas()
    existing_pairs = set(zip(sl.pred_lei, sl.succ_lei))
    df["new_signal"] = df.apply(
        lambda r: r.pred_in_gleif and r.succ_in_gleif and
                  (r.pred_lei, r.succ_lei) not in existing_pairs,
        axis=1)
    print(f"  NEW signal (both in GLEIF, not in successor_links): "
          f"{df.new_signal.sum()}", file=sys.stderr)
    print()
    print("Breakdown by property:")
    print(df[df.new_signal].source_prop.value_counts().to_string())

    out_path = os.path.join(args.data, "wikidata_lei_positives.parquet")
    df.to_parquet(out_path, compression="snappy", index=False)
    print(f"\nwrote -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
