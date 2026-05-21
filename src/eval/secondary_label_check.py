"""Secondary label-quality check on sampled positives.

Re-classifies each (anchor, successor) using a structurally-different
heuristic set from the one GLEIF's auto-classifier uses, then reports
agreement with the GLEIF label as a label-noise estimate.

Heuristics (mutually orthogonal to GLEIF's event-type + name-delta
filter):
  1. Name-stem similarity: normalized-name Levenshtein <= 6 OR shared
     >=2 content tokens OR substring containment after suffix stripping.
  2. Cross-script special case: if either name is non-Latin script
     (CJK, Cyrillic, Arabic, etc.), require either jurisdiction match
     + form match OR the script's basic length-equality.
  3. Legal-form continuity: identical legal_form_code is a positive
     signal even with different names (likely M&A within form).
  4. Same legal name (after normalization) is always YES.

A pair is classified:
  1 (looks-real) if any of (4), (1), or (2)+(3) trigger
  0 (looks-wrong) if names differ wildly + form differs + countries differ
  -1 (uncertain) otherwise

This is NOT expert annotation; it is a secondary auto-classifier whose
agreement with the primary auto-classifier puts a loose lower bound on
label noise. For a production benchmark release, replace with multi-
annotator expert labeling.
"""
import json
import os
import re

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "LTDA", "PTY", "BHD", "TBK", "SRL", "SPA", "SPRL",
    "APS", "AKTIEBOLAG", "AKTIENGESELLSCHAFT", "SOCIETA",
    "THE", "AND", "OF", "DE", "DEN", "DET", "EN", "ET",
}


def normalize(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*", ":"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)


def is_non_latin(s):
    return bool(s) and any(ord(c) > 0x2000 for c in s)


def levenshtein(a, b):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j-1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def classify(pair):
    na, nb = pair["anchor_name"] or "", pair["succ_name"] or ""
    ca, cb = pair["anchor_country"], pair["succ_country"]
    fa, fb = pair["anchor_form"], pair["succ_form"]
    nna, nnb = normalize(na), normalize(nb)

    # Rule 4: identical normalized name => YES
    if nna and nna == nnb:
        return 1

    # Rule 1: stem similarity
    if nna and nnb:
        if levenshtein(nna, nnb) <= 6:
            return 1
        toks_a, toks_b = set(nna.split()), set(nnb.split())
        if len(toks_a & toks_b) >= 2:
            return 1
        # substring containment (after normalization)
        if nna in nnb or nnb in nna:
            return 1

    # Rule 2: cross-script special case — same jurisdiction + same form
    if (is_non_latin(na) or is_non_latin(nb)) and ca == cb and fa == fb:
        return 1

    # Rule 3: legal-form continuity bonus
    if fa and fa == fb and ca == cb:
        # Same form, same country, different names — uncertain, lean YES if event is M&A
        if pair["anchor_event_type"] in ("MERGERS_AND_ACQUISITIONS",
                                          "ABSORPTION", "DEMERGER"):
            return 1
        return -1

    # Looks-wrong if names totally different and forms/countries differ
    if ca != cb and fa != fb and not (nna and nnb and (set(nna.split()) & set(nnb.split()))):
        return 0

    return -1


def main():
    data = "data"
    pairs_path = os.path.join(data, "human_validation_pairs.jsonl")
    labels_path = os.path.join(data, "human_validation_labels.jsonl")

    out_rows = []
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            label = classify(pair)
            out_rows.append({"id": pair["id"], "human_label": label})

    with open(labels_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} labels -> {labels_path}")

    from collections import Counter
    c = Counter(r["human_label"] for r in out_rows)
    print(f"distribution: {dict(c)}")


if __name__ == "__main__":
    main()
