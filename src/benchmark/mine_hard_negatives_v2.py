"""V2 hard-negative mining: replace token-overlap fuzzy fallback with
E5-small cosine top-K within the same jurisdiction.

The Phase-3 ablation showed `method_no_hard` slightly beat `method_full`;
the most likely cause is that the v1 `token_overlap_same_country` class
(232K rows, 92% of all hard negs) injects label noise. This script
replaces that fuzzy fallback with proper sentence-encoder cosine ranking.

The other three v1 classes (`parent_subsidiary`, `same_name_same_country`,
`same_name_diff_country`) are kept as-is since they're high precision.

Output: data/hard_negatives_v2.parquet
  same schema as v1's hard_negatives.parquet, but the
  `token_overlap_same_country` class is replaced by `e5_cosine_same_country`.
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
    ap.add_argument("--base-model", default="intfloat/e5-small-v2")
    ap.add_argument("--top-k", type=int, default=20,
                    help="hard negs per predecessor (per class)")
    ap.add_argument("--candidate-cap", type=int, default=20000,
                    help="cap candidates per country (largest get random subsample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
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

    # ---- load relationships for parent/subsidiary class -------------------
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

    # Per-country: subsample if too large
    keep_idx = []
    for c, sub in cand_pool.groupby("legal_addr_country"):
        if len(sub) > args.candidate_cap:
            # Always include predecessors and known successors in this country
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

    # ---- embed -----------------------------------------------------------
    print(f"loading model {args.base_model}...", file=sys.stderr)
    model = SentenceTransformer(args.base_model, device=device)
    texts = ["query: " + n for n in cand_pool["legal_name"].tolist()]
    print(f"embedding {len(texts):,} candidate names...", file=sys.stderr)
    t0 = time.time()
    embs = model.encode(texts, batch_size=128, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
    print(f"  embedded in {time.time()-t0:.1f}s -> shape {embs.shape}",
          file=sys.stderr)

    # Build per-country index
    cand_pool["embed_idx"] = np.arange(len(cand_pool))
    country_idx = {c: sub["embed_idx"].to_numpy()
                   for c, sub in cand_pool.groupby("legal_addr_country")}
    country_leis = {c: sub["lei"].to_numpy()
                    for c, sub in cand_pool.groupby("legal_addr_country")}
    lei_to_embed_idx = dict(zip(cand_pool["lei"], cand_pool["embed_idx"]))

    # ---- per-predecessor cosine top-K ------------------------------------
    print("mining E5-cosine top-K per predecessor...", file=sys.stderr)
    out_rows = []
    name_lookup = dict(zip(ent["lei"], ent["legal_name"]))
    nname_lookup = dict(zip(ent["lei"], ent["nname"]))
    t0 = time.time()
    processed = 0

    # Group predecessors by country so we can reuse the country's embedding matrix.
    pos_by_country = pos.groupby("pred_country")
    for country, sub in pos_by_country:
        country_pool_idx = country_idx.get(country)
        if country_pool_idx is None or len(country_pool_idx) < 2:
            continue
        pool_embs = embs[country_pool_idx]  # (M, d) for this country
        pool_leis = country_leis[country]   # (M,)

        for pred_lei in sub["pred_lei"].unique():
            pred_eidx = lei_to_embed_idx.get(pred_lei)
            if pred_eidx is None:
                continue
            excluded = succ_lookup.get(pred_lei, set()) | {pred_lei} | related.get(pred_lei, set())
            # cosine: pool_embs @ pred_embs.T  (normalized vectors)
            sims = pool_embs @ embs[pred_eidx]
            # filter out excluded LEIs
            mask = ~np.isin(pool_leis, list(excluded))
            sims_masked = np.where(mask, sims, -np.inf)
            # top-K
            k = min(args.top_k, len(sims_masked))
            top_local = np.argpartition(-sims_masked, k - 1)[:k]
            top_local = top_local[np.argsort(-sims_masked[top_local])]
            for j in top_local:
                if sims_masked[j] == -np.inf:
                    continue
                out_rows.append((pred_lei, str(pool_leis[j]),
                                  "e5_cosine_same_country",
                                  float(sims_masked[j]) * 100))  # score in 0-100 for consistency
            processed += 1
        if processed % 1000 == 0:
            print(f"  {processed:,} predecessors processed in {time.time()-t0:.0f}s",
                  file=sys.stderr)

    print(f"  total processed: {processed:,} in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # ---- merge with the high-precision v1 classes -------------------------
    print("merging with high-precision v1 negs (parent_subsidiary, same_name_*)...",
          file=sys.stderr)
    v1 = pq.read_table(os.path.join(args.data, "hard_negatives.parquet")).to_pandas()
    keep_classes = ["parent_subsidiary", "same_name_same_country",
                    "same_name_diff_country"]
    v1_kept = v1[v1["negative_class"].isin(keep_classes)].copy()
    v1_kept = v1_kept.rename(columns={"pred_lei": "pred_lei", "neg_lei": "neg_lei",
                                       "negative_class": "negative_class",
                                       "score": "score"})

    v2_new = pd.DataFrame(out_rows, columns=["pred_lei", "neg_lei",
                                              "negative_class", "score"])
    v2_new = v2_new.drop_duplicates(subset=["pred_lei", "neg_lei"], keep="first")
    print(f"  kept from v1: {len(v1_kept):,}", file=sys.stderr)
    print(f"  new e5_cosine negs: {len(v2_new):,}", file=sys.stderr)

    combined = pd.concat([v1_kept, v2_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pred_lei", "neg_lei"], keep="first")
    print(f"  total v2 hard negs (deduped): {len(combined):,}", file=sys.stderr)

    out_path = os.path.join(args.data, "hard_negatives_v2.parquet")
    combined.to_parquet(out_path, compression="snappy", index=False)
    print()
    print("=" * 72)
    print(f"v2 hard-neg class breakdown:")
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
