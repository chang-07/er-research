"""V3 hard-negative mining: TF-IDF char-3gram cosine within the same jurisdiction.

This script exists to test whether the v2 cross-encoder's headline numbers
depend on the choice of hard-negative miner. v2 mines with a neural encoder
(E5-small cosine); v3 mines with a structurally orthogonal signal: sparse
char-3gram TF-IDF cosine, which has no learned representation and no semantic
component. If a cross-encoder trained and evaluated on v3 produces comparable
gains over baselines, the cross-encoder approach is robust to the miner;
otherwise, the v2 headline was at least partly in-distribution.

The other three v1 classes (`parent_subsidiary`, `same_name_same_country`,
`same_name_diff_country`) are kept as-is, matching v2's design.

Output: data/hard_negatives_v3.parquet
  same schema as v2's hard_negatives_v2.parquet, but the
  `e5_cosine_same_country` class is replaced by `tfidf_char_same_country`.
"""
import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--top-k", type=int, default=20,
                    help="hard negs per predecessor (per class)")
    ap.add_argument("--candidate-cap", type=int, default=20000,
                    help="cap candidates per country (largest get random subsample)")
    ap.add_argument("--ngram", type=int, default=3,
                    help="character n-gram width for TF-IDF")
    ap.add_argument("--max-features", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- load successor links (positives) ---------------------------------
    print("loading successor links...", file=sys.stderr)
    succ = pq.read_table(
        os.path.join(args.data, "successor_links.parquet"),
        columns=["pred_lei", "succ_lei", "classification", "pred_country"],
    ).to_pandas()
    pos = succ[succ["classification"] != "likely_administrative"].copy()
    print(f"  {len(pos):,} plausibly-real positives", file=sys.stderr)
    succ_lookup = succ.groupby("pred_lei")["succ_lei"].apply(set).to_dict()

    # ---- load relationships for parent/subsidiary exclusion ---------------
    print("loading relationships...", file=sys.stderr)
    rr = pq.read_table(
        os.path.join(args.data, "relationships.parquet"),
        columns=["start_lei", "end_lei", "relationship_status"],
    ).to_pandas()
    rr = rr[rr["relationship_status"] == "ACTIVE"]
    related = defaultdict(set)
    for s, e in zip(rr["start_lei"], rr["end_lei"]):
        if s and e and s != e:
            related[s].add(e)
            related[e].add(s)

    # ---- load entities ----------------------------------------------------
    print("loading entities...", file=sys.stderr)
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country"],
    ).to_pandas().drop_duplicates(subset=["lei"], keep="first")
    ent["nname"] = ent["legal_name"].apply(normalize_name)
    ent = ent[ent["nname"].str.len() > 0].reset_index(drop=True)
    ent["legal_name"] = ent["legal_name"].fillna("")
    ent["legal_addr_country"] = ent["legal_addr_country"].fillna("")

    # ---- restrict candidate pool ------------------------------------------
    countries_with_positives = set(pos["pred_country"].dropna().unique())
    print(f"  positives span {len(countries_with_positives)} countries", file=sys.stderr)
    cand_pool = ent[ent["legal_addr_country"].isin(countries_with_positives)].copy()
    print(f"  candidate pool (same-country): {len(cand_pool):,} entities",
          file=sys.stderr)

    keep_idx = []
    for c, sub in cand_pool.groupby("legal_addr_country"):
        if len(sub) > args.candidate_cap:
            must = set(pos[pos["pred_country"] == c]["pred_lei"]) | set(
                pos[pos["pred_country"] == c]["succ_lei"])
            forced = sub[sub["lei"].isin(must)]
            other = sub[~sub["lei"].isin(must)]
            n_extra = max(0, args.candidate_cap - len(forced))
            sampled = other.sample(n=min(len(other), n_extra), random_state=args.seed)
            keep_idx.extend(forced.index.tolist())
            keep_idx.extend(sampled.index.tolist())
        else:
            keep_idx.extend(sub.index.tolist())
    cand_pool = cand_pool.loc[keep_idx].reset_index(drop=True)
    print(f"  after candidate_cap={args.candidate_cap}/country: {len(cand_pool):,}",
          file=sys.stderr)

    # ---- fit TF-IDF char-ngram over the candidate pool's legal names ------
    print(f"fitting TF-IDF char-{args.ngram}gram on {len(cand_pool):,} names...",
          file=sys.stderr)
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(args.ngram, args.ngram),
        min_df=2,
        max_features=args.max_features,
        dtype=np.float32,
        norm="l2",
    )
    cand_mat = vec.fit_transform(cand_pool["legal_name"].tolist())
    print(f"  TF-IDF matrix: {cand_mat.shape}, nnz={cand_mat.nnz:,}, "
          f"fit in {time.time()-t0:.1f}s", file=sys.stderr)

    # Per-country slice (rows of cand_mat)
    cand_pool["row_idx"] = np.arange(len(cand_pool))
    country_rows = {c: sub["row_idx"].to_numpy()
                    for c, sub in cand_pool.groupby("legal_addr_country")}
    country_leis = {c: sub["lei"].to_numpy()
                    for c, sub in cand_pool.groupby("legal_addr_country")}
    lei_to_row = dict(zip(cand_pool["lei"], cand_pool["row_idx"]))

    # ---- per-predecessor cosine top-K -------------------------------------
    print("mining TF-IDF-cosine top-K per predecessor...", file=sys.stderr)
    out_rows = []
    t0 = time.time()
    processed = 0

    pos_by_country = pos.groupby("pred_country")
    for country, sub in pos_by_country:
        country_pool_rows = country_rows.get(country)
        if country_pool_rows is None or len(country_pool_rows) < 2:
            continue
        pool_mat = cand_mat[country_pool_rows]   # (M, V) sparse
        pool_leis = country_leis[country]        # (M,)

        # Pre-compute predecessor row indices for this country
        pred_leis_uniq = sub["pred_lei"].unique()
        pred_rows = [lei_to_row.get(p) for p in pred_leis_uniq]
        valid_idx = [i for i, r in enumerate(pred_rows) if r is not None]
        if not valid_idx:
            continue
        pred_leis_valid = [pred_leis_uniq[i] for i in valid_idx]
        pred_row_idx = [pred_rows[i] for i in valid_idx]
        q_mat = cand_mat[pred_row_idx]           # (P, V) sparse

        # Cosine = q_mat @ pool_mat.T (rows already L2-normed by TfidfVectorizer)
        sims = q_mat @ pool_mat.T                # (P, M) sparse -> sparse
        sims = sims.toarray()                    # M typically <= 20k so this is OK

        for p_i, pred_lei in enumerate(pred_leis_valid):
            excluded = (succ_lookup.get(pred_lei, set())
                        | {pred_lei}
                        | related.get(pred_lei, set()))
            row_sims = sims[p_i].copy()
            # mask excluded
            mask = ~np.isin(pool_leis, list(excluded))
            row_sims = np.where(mask, row_sims, -np.inf)
            k = min(args.top_k, len(row_sims))
            top_local = np.argpartition(-row_sims, k - 1)[:k]
            top_local = top_local[np.argsort(-row_sims[top_local])]
            for j in top_local:
                if row_sims[j] == -np.inf or row_sims[j] <= 0:
                    continue
                out_rows.append((pred_lei, str(pool_leis[j]),
                                  "tfidf_char_same_country",
                                  float(row_sims[j]) * 100))
            processed += 1
        if processed and processed % 5000 < len(pred_leis_valid):
            print(f"  {processed:,} predecessors processed in {time.time()-t0:.0f}s",
                  file=sys.stderr)

    print(f"  total processed: {processed:,} in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # ---- merge with high-precision v1 classes -----------------------------
    print("merging with high-precision v1 negs (parent_subsidiary, same_name_*)...",
          file=sys.stderr)
    v1 = pq.read_table(os.path.join(args.data, "hard_negatives.parquet")).to_pandas()
    keep_classes = ["parent_subsidiary", "same_name_same_country",
                    "same_name_diff_country"]
    v1_kept = v1[v1["negative_class"].isin(keep_classes)].copy()

    v3_new = pd.DataFrame(out_rows, columns=["pred_lei", "neg_lei",
                                              "negative_class", "score"])
    v3_new = v3_new.drop_duplicates(subset=["pred_lei", "neg_lei"], keep="first")
    print(f"  kept from v1: {len(v1_kept):,}", file=sys.stderr)
    print(f"  new tfidf_char negs: {len(v3_new):,}", file=sys.stderr)

    combined = pd.concat([v1_kept, v3_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pred_lei", "neg_lei"], keep="first")
    print(f"  total v3 hard negs (deduped): {len(combined):,}", file=sys.stderr)

    out_path = os.path.join(args.data, "hard_negatives_v3.parquet")
    combined.to_parquet(out_path, compression="snappy", index=False)
    print()
    print("=" * 72)
    print(f"v3 hard-neg class breakdown:")
    print(combined["negative_class"].value_counts().to_string())
    print()
    counts = combined.groupby("pred_lei").size()
    print(f"Negatives per predecessor (after dedupe):")
    print(f"  min={counts.min()}  median={int(counts.median())}  "
          f"mean={counts.mean():.1f}  max={counts.max()}  "
          f"n_preds_with_any={len(counts):,}")
    print(f"wrote -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
