"""Pull high-profile company successor pairs from Wikidata, match to GLEIF.

Wikidata properties used:
  P156  followed by   (the entity's successor)
  P155  follows        (the entity's predecessor)
  P127  owned by
  P749  parent organization

This is intentionally a marquee / qualitative supplement — Wikidata's company
M&A coverage is biased toward famous cases (Twitter -> X, Facebook -> Meta,
HP split, etc.). It's useful for paper figures, sanity checks, and as a
held-out "famous cases" eval slice.

Output: data/wikidata_marquee.parquet
  wd_pred, wd_pred_label, wd_succ, wd_succ_label,
  gleif_pred_lei, gleif_succ_lei,
  gleif_pred_country, gleif_succ_country,
  match_quality   (both, pred_only, succ_only, none)
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

SPARQL_URL = "https://query.wikidata.org/sparql"
UA = "er-research/0.1 admin@chang-07.dev"

# Restrict to entities that are (subclass-of) business/organization to avoid
# pulling people, films, etc.
SUCCESSOR_QUERY = """
SELECT DISTINCT ?company ?companyLabel ?successor ?successorLabel
                ?country ?countryLabel ?succCountry ?succCountryLabel
                ?dissolved
WHERE {
  ?company wdt:P31/wdt:P279* wd:Q4830453 .      # instance of (sub)business
  ?company wdt:P156 ?successor .                 # followed by
  ?successor wdt:P31/wdt:P279* wd:Q4830453 .
  OPTIONAL { ?company  wdt:P17 ?country . }
  OPTIONAL { ?successor wdt:P17 ?succCountry . }
  OPTIONAL { ?company  wdt:P746 ?dissolved . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,fr,de,es,ja,zh". }
}
LIMIT 8000
"""

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "LTDA", "PTY", "BHD", "TBK", "SRL", "SPA", "SPRL",
    "THE", "AND", "OF",
}

# Wikidata country QIDs to ISO 3166 alpha-2 (only the ones we care to match)
WD_COUNTRY_TO_ISO = {
    "Q30": "US", "Q145": "GB", "Q183": "DE", "Q142": "FR", "Q38": "IT",
    "Q29": "ES", "Q17": "JP", "Q148": "CN", "Q865": "TW", "Q334": "SG",
    "Q155": "BR", "Q16": "CA", "Q408": "AU", "Q39": "CH", "Q55": "NL",
    "Q668": "IN", "Q31": "BE", "Q34": "SE", "Q35": "DK", "Q33": "FI",
    "Q20": "NO", "Q40": "AT", "Q403": "RS", "Q43": "TR", "Q794": "IR",
    "Q884": "KR",
}

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)

def fetch_sparql(query):
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(SPARQL_URL, data=data,
                                  headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def extract_qid(uri):
    # http://www.wikidata.org/entity/Q42 -> Q42
    return uri.rsplit("/", 1)[-1] if uri else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    print(f"Querying Wikidata SPARQL...", file=sys.stderr)
    try:
        result = fetch_sparql(SUCCESSOR_QUERY)
    except Exception as e:
        print(f"SPARQL fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    bindings = result["results"]["bindings"]
    print(f"  got {len(bindings):,} bindings", file=sys.stderr)

    rows = []
    for b in bindings:
        rows.append({
            "wd_pred": extract_qid(b.get("company", {}).get("value")),
            "wd_pred_label": b.get("companyLabel", {}).get("value"),
            "wd_succ": extract_qid(b.get("successor", {}).get("value")),
            "wd_succ_label": b.get("successorLabel", {}).get("value"),
            "wd_pred_country_qid": extract_qid(b.get("country", {}).get("value")),
            "wd_succ_country_qid": extract_qid(b.get("succCountry", {}).get("value")),
        })
    wd = pd.DataFrame(rows).drop_duplicates(subset=["wd_pred", "wd_succ"])
    # Drop bindings where the label is just the QID (no English label)
    wd = wd[~wd["wd_pred_label"].str.startswith("Q", na=False)]
    wd = wd[~wd["wd_succ_label"].str.startswith("Q", na=False)]
    print(f"  {len(wd):,} unique labeled pred->succ pairs", file=sys.stderr)

    wd["wd_pred_country"] = wd["wd_pred_country_qid"].map(WD_COUNTRY_TO_ISO)
    wd["wd_succ_country"] = wd["wd_succ_country_qid"].map(WD_COUNTRY_TO_ISO)

    # Load GLEIF entity name -> LEI index (build a (normalized_name, country) lookup)
    print("Loading GLEIF entities for name matching...", file=sys.stderr)
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country"],
    ).to_pandas()
    ent["nname"] = ent["legal_name"].apply(normalize_name)
    ent = ent[ent["nname"].str.len() > 0]

    # Build (nname, country) -> LEI; if many, pick first (could be improved)
    name_country = defaultdict(list)
    name_only = defaultdict(list)
    for lei, nname, country in zip(ent["lei"], ent["nname"], ent["legal_addr_country"].fillna("")):
        name_country[(nname, country)].append(lei)
        name_only[nname].append((lei, country))

    def lookup(label, country):
        if not label:
            return None, None
        nn = normalize_name(label)
        if not nn:
            return None, None
        if country and (nn, country) in name_country:
            leis = name_country[(nn, country)]
            return leis[0], country
        # fall back to first match in any country
        if nn in name_only:
            lei, c = name_only[nn][0]
            return lei, c
        return None, None

    print("Matching Wikidata pairs to GLEIF...", file=sys.stderr)
    pred_leis, pred_countries = [], []
    succ_leis, succ_countries = [], []
    for _, r in wd.iterrows():
        pl, pc = lookup(r["wd_pred_label"], r["wd_pred_country"])
        sl, sc = lookup(r["wd_succ_label"], r["wd_succ_country"])
        pred_leis.append(pl)
        pred_countries.append(pc)
        succ_leis.append(sl)
        succ_countries.append(sc)
    wd["gleif_pred_lei"] = pred_leis
    wd["gleif_pred_country"] = pred_countries
    wd["gleif_succ_lei"] = succ_leis
    wd["gleif_succ_country"] = succ_countries

    def quality(row):
        p = row["gleif_pred_lei"] is not None
        s = row["gleif_succ_lei"] is not None
        if p and s: return "both"
        if p: return "pred_only"
        if s: return "succ_only"
        return "none"
    wd["match_quality"] = wd.apply(quality, axis=1)

    print()
    print(f"Match quality breakdown:")
    print(wd["match_quality"].value_counts().to_string())
    print()
    both = wd[wd["match_quality"] == "both"]
    print(f"Pairs with BOTH sides mapped to GLEIF: {len(both):,}")
    print()
    print("Sample 'both' matches (Wikidata label -> GLEIF LEI):")
    sample = both.head(15)
    for _, r in sample.iterrows():
        print(f"  '{r['wd_pred_label']:<35}' -> '{r['wd_succ_label']:<35}'  "
              f"({r['gleif_pred_lei']} -> {r['gleif_succ_lei']})")

    out = os.path.join(args.data, "wikidata_marquee.parquet")
    keep = ["wd_pred", "wd_pred_label", "wd_succ", "wd_succ_label",
            "wd_pred_country", "wd_succ_country",
            "gleif_pred_lei", "gleif_pred_country",
            "gleif_succ_lei", "gleif_succ_country",
            "match_quality"]
    wd[keep].to_parquet(out, compression="snappy", index=False)
    print()
    print(f"wrote -> {out}  ({os.path.getsize(out)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
