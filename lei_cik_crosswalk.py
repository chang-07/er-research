"""LEI <-> CIK crosswalk feasibility spike.

Fetches SEC's company_tickers.json, normalizes legal names against GLEIF's
US-jurisdiction entities, and reports how many can be confidently linked.

This answers Phase-0 question 2: "can we reliably map LEI <-> CIK for ~1000
US public companies?" If yes, we can use EDGAR 8-K Item 5.03 to enrich
labels. If no, we drop EDGAR as a supplement.
"""
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from lxml import etree

PATH = "/Users/chang/Personal/research/20260519-gleif-concatenated-file-lei2.xml"
OUT_DIR = "/Users/chang/Personal/research/data"
NS = "http://www.gleif.org/data/schema/leidata/2016"
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC requires a UA containing a contact email; see
# https://www.sec.gov/os/accessing-edgar-data
UA = "er-research/0.1 admin@chang-07.dev"

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "THE", "AND",
}

def tag(name):
    return f"{{{NS}}}{name}"

def clear(rec):
    rec.clear()
    while rec.getprevious() is not None:
        del rec.getparent()[0]

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"']:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)

def main():
    # 1. Fetch SEC tickers
    print(f"Fetching {SEC_URL}...", file=sys.stderr, flush=True)
    req = urllib.request.Request(SEC_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        sec = json.load(r)
    sec_records = list(sec.values())
    print(f"  {len(sec_records):,} SEC ticker records", file=sys.stderr)

    sec_index = {}
    for rec in sec_records:
        nname = normalize_name(rec["title"])
        if nname:
            sec_index.setdefault(nname, []).append(rec)
    print(f"  {len(sec_index):,} unique normalized names "
          f"(collision rate: {1 - len(sec_index)/len(sec_records):.1%})",
          file=sys.stderr)

    # Save SEC tickers for later use
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sec_company_tickers.json"), "w") as f:
        json.dump(sec, f)

    # 2. Stream GLEIF US entities, attempt matches
    print("Scanning GLEIF US-jurisdiction entities...", file=sys.stderr, flush=True)
    t0 = time.time()
    total_us = 0
    unique_match = 0
    ambiguous_match = 0
    no_match = 0
    sample_unique = []
    sample_ambig = []
    sample_nomatch_publicish = []

    public_hints = ("INC", "INC.", "CORP", "CORP.", "CORPORATION",
                    "COMPANY", "PLC", "HOLDINGS", "GROUP")

    ctx = etree.iterparse(PATH, tag=tag("LEIRecord"), events=("end",))
    for _, rec in ctx:
        entity = rec.find(tag("Entity"))
        if entity is None:
            clear(rec); continue
        addr = entity.find(tag("LegalAddress"))
        country = (addr.findtext(tag("Country")) if addr is not None else "") or ""
        if country != "US":
            clear(rec); continue

        total_us += 1
        lei = (rec.findtext(tag("LEI")) or "").strip()
        name = (entity.findtext(tag("LegalName")) or "").strip()
        nname = normalize_name(name)
        matches = sec_index.get(nname, []) if nname else []

        if len(matches) == 1:
            unique_match += 1
            if len(sample_unique) < 15:
                m = matches[0]
                sample_unique.append((lei, name, m["cik_str"], m["ticker"], m["title"]))
        elif len(matches) > 1:
            ambiguous_match += 1
            if len(sample_ambig) < 10:
                sample_ambig.append((lei, name, [(m["cik_str"], m["ticker"]) for m in matches[:3]]))
        else:
            no_match += 1
            if len(sample_nomatch_publicish) < 15 and any(name.endswith(s) for s in public_hints):
                sample_nomatch_publicish.append((lei, name))
        clear(rec)

    elapsed = time.time() - t0
    print(f"  scanned in {elapsed:.1f}s", file=sys.stderr)

    print()
    print("=" * 72)
    print("LEI <-> CIK CROSSWALK SPIKE")
    print("=" * 72)
    print(f"GLEIF entities with US jurisdiction       : {total_us:,}")
    print(f"SEC ticker records                        : {len(sec_records):,}")
    print(f"  unique normalized SEC names             : {len(sec_index):,}")
    print()
    print(f"Unique name -> single CIK (confident link): "
          f"{unique_match:,}  ({100*unique_match/total_us:.2f}% of US GLEIF)")
    print(f"Ambiguous (1 name -> >1 CIK)              : "
          f"{ambiguous_match:,}  ({100*ambiguous_match/total_us:.2f}%)")
    print(f"No match                                  : "
          f"{no_match:,}  ({100*no_match/total_us:.2f}%)")
    print()
    coverage = unique_match / len(sec_index) if sec_index else 0
    print(f"% of SEC filers we found in GLEIF (US)    : ~{100*coverage:.1f}%")
    print()
    print("Sample unique matches (GLEIF -> SEC):")
    for lei, name, cik, ticker, title in sample_unique:
        print(f"  {lei}  '{name[:35]:35s}' -> CIK {cik:>8}  {ticker:>6}  '{title[:35]}'")
    print()
    print("Sample ambiguous (multiple CIKs for one normalized name):")
    for lei, name, ms in sample_ambig:
        print(f"  '{name[:40]:40s}' -> {ms}")
    print()
    print("Sample 'looks public-ish' no-matches:")
    print("  (subsidiaries / private cos / DBA names — expected; need OpenFIGI fallback)")
    for lei, name in sample_nomatch_publicish:
        print(f"  {lei}  '{name}'")
    print()
    print("=" * 72)
    print("VERDICT (spike target: >=1000 confident matches)")
    print("=" * 72)
    if unique_match >= 1000:
        print(f">> SPIKE PASS. {unique_match:,} confident GLEIF <-> CIK matches via exact-name join.")
        print("   Sufficient to use EDGAR 8-K Item 5.03 as a supplementary label source for US.")
    elif unique_match >= 500:
        print(f">> SPIKE CAUTION. {unique_match:,} matches. Add OpenFIGI or fuzzy match for scale.")
    else:
        print(f">> SPIKE WEAK. {unique_match:,} matches. Reconsider EDGAR as primary supplement.")

if __name__ == "__main__":
    main()
