"""Mine hard negatives for each positive successor pair.

For each predecessor LEI (one side of a real successor link), generate
candidate "negative" LEIs — entities that look similar by some signal but
should NOT be matched as the same legal-economic entity. These force the
model to learn the lifecycle-vs-similarity distinction.

Four negative classes:

  parent_subsidiary
      From relationships.parquet: any LEI that is a parent / subsidiary of
      the predecessor (direct or ultimate consolidation, fund mgmt, subfund,
      international branch). These look related but are legally distinct.

  same_name_same_country
      Distinct LEI with the same normalized legal name in the same country.
      Highest-precision name collisions.

  same_name_diff_country
      Distinct LEI with the same normalized legal name in a different
      country. (Many GLEIF entries share generic names across jurisdictions.)

  token_overlap_same_country
      Distinct LEI in the same country whose normalized name shares >=2
      content tokens with the predecessor's normalized name. Fuzzy fallback
      to ensure we get enough negs per pos when exact name collisions are
      sparse.

Output: data/hard_negatives.parquet
  pred_lei, neg_lei, negative_class, score
where score is in [0, 100]; 100 means exact normalized-name match or a
documented relationship, lower means weaker fuzzy match.

The actual (anchor, candidate, label) training triples are assembled by a
later step (build_splits.py) that joins hard_negatives.parquet with the
positives and applies the train/test ratios.
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "LTDA", "PTY", "BHD", "TBK", "SRL", "SPA", "SPRL",
    "THE", "AND", "OF",
}

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)

def tokens(nname):
    return set(nname.split()) if nname else set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-per-class", type=int, default=20,
                    help="cap negatives per (predecessor x class)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Loading entities...", file=sys.stderr, flush=True)
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country"],
    ).to_pandas()
    ent["nname"] = ent["legal_name"].apply(normalize_name)
    ent = ent[ent["nname"].str.len() > 0].reset_index(drop=True)
    print(f"  {len(ent):,} entities with non-empty normalized name", file=sys.stderr)

    print("Loading successor links...", file=sys.stderr, flush=True)
    succ = pq.read_table(
        os.path.join(args.data, "successor_links.parquet"),
        columns=["pred_lei", "succ_lei", "classification"],
    ).to_pandas()
    # Drop likely_administrative — we don't generate negs for those
    succ_clean = succ[succ["classification"] != "likely_administrative"].copy()
    print(f"  {len(succ_clean):,} usable positives (filtered out "
          f"{len(succ) - len(succ_clean)} likely_administrative)", file=sys.stderr)

    # Exclusion set: any LEI that is on the successor side of a known pair —
    # never use those as negatives for the corresponding predecessor.
    succ_lookup = succ.groupby("pred_lei")["succ_lei"].apply(set).to_dict()

    print("Loading relationships...", file=sys.stderr, flush=True)
    rr = pq.read_table(
        os.path.join(args.data, "relationships.parquet"),
        columns=["start_lei", "end_lei", "relationship_type", "relationship_status"],
    ).to_pandas()
    rr = rr[rr["relationship_status"] == "ACTIVE"]
    # Build symmetric LEI -> related set
    related = defaultdict(set)
    for s, e in zip(rr["start_lei"], rr["end_lei"]):
        if s and e and s != e:
            related[s].add(e)
            related[e].add(s)
    print(f"  {sum(len(v) for v in related.values()):,} symmetric "
          f"relationship edges across {len(related):,} LEIs", file=sys.stderr)

    print("Building name indices...", file=sys.stderr, flush=True)
    # (nname, country) -> list of LEIs
    name_country_idx = defaultdict(list)
    # nname -> list of (lei, country)
    name_idx = defaultdict(list)
    # token -> set of LEIs (for fuzzy fallback)
    token_idx = defaultdict(set)
    # lei -> (nname, country, token_set)
    lei_info = {}

    for lei, name, country, nname in zip(ent["lei"], ent["legal_name"],
                                          ent["legal_addr_country"], ent["nname"]):
        country = country or ""
        toks = tokens(nname)
        lei_info[lei] = (nname, country, toks)
        name_country_idx[(nname, country)].append(lei)
        name_idx[nname].append((lei, country))
        # only index moderately-rare tokens for fuzzy fallback (skip ultra-common)
        for t in toks:
            if len(t) >= 4:
                token_idx[t].add(lei)
    print(f"  {len(name_country_idx):,} (nname, country) buckets, "
          f"{len(token_idx):,} fuzzy tokens", file=sys.stderr)

    # Mine negatives
    print("Mining hard negatives...", file=sys.stderr, flush=True)
    t0 = time.time()
    out_rows = []
    n_with_any = 0
    class_totals = defaultdict(int)

    for i, pred_lei in enumerate(succ_clean["pred_lei"].unique()):
        if pred_lei not in lei_info:
            continue
        nname, country, toks = lei_info[pred_lei]
        if not nname:
            continue

        excluded = succ_lookup.get(pred_lei, set()) | {pred_lei}

        # Class 1: parent_subsidiary (highest precision)
        c1 = [x for x in related.get(pred_lei, []) if x not in excluded]
        for neg in c1[:args.max_per_class]:
            out_rows.append((pred_lei, neg, "parent_subsidiary", 100))
            class_totals["parent_subsidiary"] += 1

        # Class 2: same_name_same_country
        same = [x for x in name_country_idx.get((nname, country), []) if x not in excluded]
        for neg in same[:args.max_per_class]:
            out_rows.append((pred_lei, neg, "same_name_same_country", 100))
            class_totals["same_name_same_country"] += 1

        # Class 3: same_name_diff_country
        diff = [lei for lei, c in name_idx.get(nname, [])
                if lei not in excluded and c != country]
        for neg in diff[:args.max_per_class]:
            out_rows.append((pred_lei, neg, "same_name_diff_country", 100))
            class_totals["same_name_diff_country"] += 1

        # Class 4: fuzzy fallback — same country, ≥2 shared content tokens
        # (only run if classes 1-3 didn't yield enough)
        already_for_this_pred = sum(1 for r in out_rows[-(3*args.max_per_class):]
                                    if r[0] == pred_lei)
        if already_for_this_pred < 5 and len(toks) >= 2:
            candidate_leis = set()
            for t in toks:
                if len(t) >= 4:
                    candidate_leis.update(token_idx.get(t, set()))
            candidate_leis -= excluded
            scored = []
            for cand in candidate_leis:
                if cand not in lei_info:
                    continue
                c_nname, c_country, c_toks = lei_info[cand]
                if c_country != country:
                    continue
                if not c_toks:
                    continue
                # Jaccard token overlap
                overlap = len(toks & c_toks)
                if overlap >= 2:
                    score = int(100 * overlap / max(len(toks), len(c_toks)))
                    if score >= 50:
                        scored.append((cand, score))
            scored.sort(key=lambda x: -x[1])
            for cand, score in scored[:args.max_per_class]:
                out_rows.append((pred_lei, cand, "token_overlap_same_country", score))
                class_totals["token_overlap_same_country"] += 1

        # Track per-predecessor success
        any_for_pred = (len(c1) > 0 or len(same) > 0 or len(diff) > 0
                        or already_for_this_pred > 0)
        if any_for_pred:
            n_with_any += 1

        if (i + 1) % 5000 == 0:
            print(f"  [{time.time()-t0:5.1f}s] processed {i+1:,} predecessors  "
                  f"({len(out_rows):,} negs so far)", file=sys.stderr, flush=True)

    df = pd.DataFrame(out_rows, columns=["pred_lei", "neg_lei", "negative_class", "score"])
    df = df.drop_duplicates(subset=["pred_lei", "neg_lei"], keep="first")

    print()
    print("=" * 72)
    print(f"Total positives processed         : {len(succ_clean['pred_lei'].unique()):,}")
    print(f"Predecessors with >=1 hard neg    : {n_with_any:,}")
    print(f"Total (pred, neg) pairs (deduped) : {len(df):,}")
    print()
    print("Per-class totals (before dedupe):")
    for c, n in sorted(class_totals.items(), key=lambda x: -x[1]):
        print(f"  {c:32s} {n:>8,}")
    print()
    print("After dedupe:")
    print(df["negative_class"].value_counts().to_string())
    print()

    # Per-predecessor negative count distribution
    counts = df.groupby("pred_lei").size()
    print(f"Negatives per predecessor (after dedupe):")
    print(f"  min={counts.min()}  median={int(counts.median())}  "
          f"mean={counts.mean():.1f}  max={counts.max()}")
    print()
    out = os.path.join(args.data, "hard_negatives.parquet")
    df.to_parquet(out, compression="snappy", index=False)
    print(f"wrote -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
