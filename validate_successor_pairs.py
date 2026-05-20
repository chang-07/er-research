"""Validate label quality of GLEIF SuccessorEntity links.

For each pair (predecessor LEI -> successor LEI), compares the two entities'
Level-1 records and classifies the pair by how plausibly it represents a real
lifecycle event vs. an administrative re-issuance.

Outputs:
  data/successor_validation_sample.csv  — 100 random pairs with all features
                                          for manual spot-check
Prints aggregate classification stats.
"""
import csv
import os
import random
import sys
import time
from collections import Counter
from lxml import etree

PATH = "/Users/chang/Personal/research/20260519-gleif-concatenated-file-lei2.xml"
OUT_DIR = "/Users/chang/Personal/research/data"
NS = "http://www.gleif.org/data/schema/leidata/2016"

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

def tag(name):
    return f"{{{NS}}}{name}"

def clear(rec):
    rec.clear()
    while rec.getprevious() is not None:
        del rec.getparent()[0]

def extract_summary(entity):
    addr = entity.find(tag("LegalAddress"))
    legal_form = entity.find(tag("LegalForm"))
    return {
        "name": (entity.findtext(tag("LegalName")) or "").strip(),
        "country": (addr.findtext(tag("Country")) if addr is not None else "") or "",
        "city": (addr.findtext(tag("City")) if addr is not None else "") or "",
        "form": (legal_form.findtext(tag("EntityLegalFormCode")) if legal_form is not None else "") or "",
        "status": (entity.findtext(tag("EntityStatus")) or "").strip(),
    }

def extract_events(entity):
    events = entity.find(tag("LegalEntityEvents"))
    if events is None:
        return []
    out = []
    for ev in events.findall(tag("LegalEntityEvent")):
        if ev.get("event_status") != "COMPLETED":
            continue
        etype = ev.findtext(tag("LegalEntityEventType"))
        eff = ev.findtext(tag("LegalEntityEventEffectiveDate"))
        if etype:
            out.append((etype, eff or ""))
    return out

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"']:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)

def classify(pred, succ, pred_events):
    if succ is None:
        return "successor_not_found"
    same_name = normalize_name(pred["name"]) == normalize_name(succ["name"])
    same_country = pred["country"] == succ["country"]
    same_form = pred["form"] == succ["form"]
    has_real_event = any(t in REAL_EVENT_TYPES for t, _ in pred_events)
    if has_real_event and not same_name:
        return "strong_real"
    if has_real_event and same_name:
        return "likely_real_event_only"
    if not same_name:
        return "likely_real_name_changed"
    if same_name and same_country and same_form:
        return "likely_administrative"
    return "ambiguous"

def main():
    # Pass 1: find successor pairs and store predecessor summaries
    print("Pass 1: collecting SuccessorEntity links...", file=sys.stderr, flush=True)
    t0 = time.time()
    pairs = []
    want_succ = set()
    ctx = etree.iterparse(PATH, tag=tag("LEIRecord"), events=("end",))
    n = 0
    for _, rec in ctx:
        n += 1
        entity = rec.find(tag("Entity"))
        if entity is not None:
            succ = entity.find(tag("SuccessorEntity"))
            if succ is not None:
                succ_lei = (succ.findtext(tag("SuccessorLEI")) or "").strip()
                if succ_lei:
                    pairs.append({
                        "pred_lei": (rec.findtext(tag("LEI")) or "").strip(),
                        "succ_lei": succ_lei,
                        "pred": extract_summary(entity),
                        "events": extract_events(entity),
                    })
                    want_succ.add(succ_lei)
        clear(rec)
    print(f"  scanned {n:,} records in {time.time()-t0:.1f}s, "
          f"found {len(pairs):,} pairs ({len(want_succ):,} unique successor LEIs)",
          file=sys.stderr)

    # Pass 2: resolve successor LEIs
    print("Pass 2: resolving successor records...", file=sys.stderr, flush=True)
    t1 = time.time()
    succ_records = {}
    ctx = etree.iterparse(PATH, tag=tag("LEIRecord"), events=("end",))
    for _, rec in ctx:
        lei = (rec.findtext(tag("LEI")) or "").strip()
        if lei in want_succ:
            entity = rec.find(tag("Entity"))
            if entity is not None:
                succ_records[lei] = extract_summary(entity)
        clear(rec)
    print(f"  resolved {len(succ_records):,}/{len(want_succ):,} successors in "
          f"{time.time()-t1:.1f}s", file=sys.stderr)

    # Classify
    classification = Counter()
    enriched = []
    for p in pairs:
        succ = succ_records.get(p["succ_lei"])
        klass = classify(p["pred"], succ, p["events"])
        classification[klass] += 1
        enriched.append((p, succ, klass))

    # Stats
    print()
    print("=" * 72)
    print("LABEL QUALITY AUDIT — SuccessorEntity pairs")
    print("=" * 72)
    print(f"Total pairs                       : {len(pairs):,}")
    print(f"Successors resolved (in dataset)  : {len(succ_records):,}  "
          f"({100*len(succ_records)/len(want_succ):.1f}%)")
    print()
    print("Classification:")
    print(f"  {'CLASS':32s} {'COUNT':>8s}  {'%':>6s}")
    for k, c in classification.most_common():
        print(f"  {k:32s} {c:>8,}  {100*c/len(pairs):>5.2f}%")
    print()

    # The headline number we care about
    plausibly_real = sum(classification[k] for k in
        ("strong_real", "likely_real_name_changed", "likely_real_event_only", "ambiguous"))
    admin_or_missing = sum(classification[k] for k in
        ("likely_administrative", "successor_not_found"))
    print(f"Plausibly-real pairs (usable as labels)         : "
          f"{plausibly_real:,}  ({100*plausibly_real/len(pairs):.2f}%)")
    print(f"Administrative or unresolved (filter out)       : "
          f"{admin_or_missing:,}  ({100*admin_or_missing/len(pairs):.2f}%)")
    print()
    print("GATE: paper viability requires >=80% plausibly-real.")
    print(f"   ACTUAL: {100*plausibly_real/len(pairs):.2f}%  -> "
          f"{'PASS' if plausibly_real/len(pairs) >= 0.80 else 'NEEDS FILTERING'}")
    print()

    # Sample 100 random pairs to CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "successor_validation_sample.csv")
    random.seed(42)
    sample = random.sample(enriched, min(100, len(enriched)))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "classification", "pred_lei", "succ_lei",
            "pred_name", "succ_name",
            "pred_country", "succ_country",
            "pred_form", "succ_form",
            "pred_status", "succ_status",
            "n_pred_events", "events",
        ])
        for p, s, k in sample:
            ev_str = "; ".join(f"{t}@{d[:10]}" for t, d in p["events"]) if p["events"] else ""
            w.writerow([
                k, p["pred_lei"], p["succ_lei"],
                p["pred"]["name"], (s["name"] if s else ""),
                p["pred"]["country"], (s["country"] if s else ""),
                p["pred"]["form"], (s["form"] if s else ""),
                p["pred"]["status"], (s["status"] if s else ""),
                len(p["events"]), ev_str,
            ])
    print(f"Wrote {len(sample)} samples for spot-check -> {out_path}")

if __name__ == "__main__":
    main()
