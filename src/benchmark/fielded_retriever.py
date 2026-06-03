"""Build a fielded-retriever candidate pool.

Addresses the limitation in §6 that all TemporalLink candidates so far
come from the same family of miners (E5-cosine, char-3gram). A truly
fielded ER pipeline retrieves candidates from the FULL entity universe
using a bi-encoder, then reranks the top-K with a cross-encoder. This
script simulates that:

  1. Encode all 3.3M entities with the trained method_v2_full bi-encoder
     (one-time cost, cached to disk).
  2. For each test anchor (TIME/EVENT/JURIS), retrieve top-K candidates
     by cosine similarity. Default retrieval pool: same-country only
     (most realistic for a deployment that already has light filtering).
  3. Build a new pair file with the retrieved candidates, labelling
     each pair as positive if (anchor, candidate) appears in successor_links.
  4. Output: data/pairs_fielded.parquet — drop-in for evaluate.py.

The downstream cross-encoder + LLM baseline scoring is done separately
with the existing score_pairs / cross-encoder predict pipeline.

Usage:
  python3 src/benchmark/fielded_retriever.py --model data/models/method_v2_full --k 20
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def format_entity(name, country, form):
    name = (name or "").strip() or "(unknown)"
    country = (country or "??").strip() or "??"
    form = (form or "??").strip() or "??"
    return f"query: {name} | jurisdiction: {country} | form: {form}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="data/models/method_v2_full",
                    help="path to a saved sentence-transformers bi-encoder")
    ap.add_argument("--k", type=int, default=20,
                    help="top-K candidates to retrieve per anchor")
    ap.add_argument("--restrict-country", action="store_true", default=True,
                    help="restrict retrieval pool to same-country entities "
                         "(realistic light filtering before semantic retrieval)")
    ap.add_argument("--no-restrict-country", dest="restrict_country",
                    action="store_false")
    ap.add_argument("--embedding-cache",
                    default="data/entity_embeddings_v2_full.npy")
    ap.add_argument("--out", default="data/pairs_fielded.parquet")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    # ---- load all entities --------------------------------------------------
    print("loading entities.parquet...", file=sys.stderr)
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas().drop_duplicates("lei", keep="first").reset_index(drop=True)
    print(f"  {len(ent):,} unique entities", file=sys.stderr)

    # Pre-format text for every entity
    ent["text"] = [format_entity(n, c, f) for n, c, f in
                    zip(ent["legal_name"], ent["legal_addr_country"],
                        ent["legal_form_code"])]

    # ---- encode (or load cached embeddings) ---------------------------------
    if os.path.exists(args.embedding_cache):
        print(f"loading cached embeddings from {args.embedding_cache}",
              file=sys.stderr)
        embeds = np.load(args.embedding_cache)
        assert len(embeds) == len(ent), \
            f"cached embed count {len(embeds)} != entity count {len(ent)}"
    else:
        print(f"loading bi-encoder from {args.model}...", file=sys.stderr)
        model = SentenceTransformer(args.model, device=device)
        print(f"encoding {len(ent):,} entities (batch={args.batch_size})...",
              file=sys.stderr)
        t0 = time.time()
        embeds = model.encode(ent["text"].tolist(),
                                batch_size=args.batch_size,
                                show_progress_bar=True,
                                convert_to_numpy=True,
                                normalize_embeddings=True)
        embeds = embeds.astype(np.float16)  # halve memory
        print(f"  encoded in {time.time()-t0:.0f}s; shape {embeds.shape}",
              file=sys.stderr)
        np.save(args.embedding_cache, embeds)
        print(f"  cached to {args.embedding_cache}", file=sys.stderr)

    # Build LEI -> row-index lookup
    lei_to_idx = {lei: i for i, lei in enumerate(ent["lei"].values)}

    # ---- load test anchors (union across 3 splits) --------------------------
    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label",
                  "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()
    is_test_pos = ((pairs["split_time"] == "test")
                    | (pairs["split_event"] == "test")
                    | (pairs["split_jurisdiction"] == "test")) & (pairs["label"] == 1)
    test_anchors_with_truth = (pairs[is_test_pos]
                                .groupby("anchor_lei")
                                .agg({"candidate_lei": list,
                                       "split_time": "first",
                                       "split_event": "first",
                                       "split_jurisdiction": "first"})
                                .reset_index())
    print(f"\n{len(test_anchors_with_truth):,} unique test anchors with positives",
          file=sys.stderr)

    # ---- retrieve top-K per anchor ------------------------------------------
    out_rows = []
    pool_by_country = None
    if args.restrict_country:
        # Build country -> entity row-indices once
        pool_by_country = {}
        for i, c in enumerate(ent["legal_addr_country"].values):
            pool_by_country.setdefault(c or "??", []).append(i)
        sizes = sorted([len(v) for v in pool_by_country.values()], reverse=True)
        print(f"same-country pools: top-5 sizes {sizes[:5]}, "
              f"min {sizes[-1]}, total countries {len(pool_by_country)}",
              file=sys.stderr)

    embeds_f32 = embeds.astype(np.float32)   # cosine search in float32
    embeds_norm = np.linalg.norm(embeds_f32, axis=1, keepdims=True)
    embeds_unit = embeds_f32 / np.clip(embeds_norm, 1e-8, None)

    t0 = time.time()
    n_seen = 0
    for _, row in test_anchors_with_truth.iterrows():
        anchor_lei = row["anchor_lei"]
        anchor_idx = lei_to_idx.get(anchor_lei)
        if anchor_idx is None:
            continue
        anchor_country = ent.iloc[anchor_idx]["legal_addr_country"] or "??"
        if args.restrict_country:
            pool = pool_by_country.get(anchor_country, [])
            if len(pool) < 2:
                # Fall back to full pool if country has too few entities
                pool = list(range(len(ent)))
            pool_arr = np.asarray(pool, dtype=np.int64)
            sims = embeds_unit[pool_arr] @ embeds_unit[anchor_idx]
        else:
            pool_arr = np.arange(len(ent), dtype=np.int64)
            sims = embeds_unit @ embeds_unit[anchor_idx]

        # exclude the anchor itself
        anchor_in_pool = np.where(pool_arr == anchor_idx)[0]
        if len(anchor_in_pool):
            sims[anchor_in_pool] = -1.0
        # top-K
        k = min(args.k, len(pool))
        if k == 0:
            continue
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        top_global = pool_arr[top_local]

        truth_set = set(row["candidate_lei"])
        for ti, global_idx in enumerate(top_global):
            cand_lei = ent.iloc[int(global_idx)]["lei"]
            out_rows.append({
                "anchor_lei": anchor_lei,
                "candidate_lei": cand_lei,
                "label": 1 if cand_lei in truth_set else 0,
                "source": "fielded_retriever_top20",
                "retrieval_rank": ti + 1,
                "retrieval_score": float(sims[top_local[ti]]),
                "split_time": row["split_time"],
                "split_event": row["split_event"],
                "split_jurisdiction": row["split_jurisdiction"],
            })

        # Also include any TRUE positives not retrieved in top-K so we can
        # measure recall@K honestly
        for true_succ in truth_set:
            if true_succ not in set(ent.iloc[top_global]["lei"]):
                succ_idx = lei_to_idx.get(true_succ)
                if succ_idx is None:
                    continue
                if args.restrict_country:
                    in_pool = succ_idx in pool
                else:
                    in_pool = True
                # Compute its similarity even if outside top-K, for diagnostics
                sim = float(embeds_unit[succ_idx] @ embeds_unit[anchor_idx])
                out_rows.append({
                    "anchor_lei": anchor_lei,
                    "candidate_lei": true_succ,
                    "label": 1,
                    "source": "fielded_retriever_true_missed",
                    "retrieval_rank": -1,
                    "retrieval_score": sim,
                    "split_time": row["split_time"],
                    "split_event": row["split_event"],
                    "split_jurisdiction": row["split_jurisdiction"],
                })

        n_seen += 1
        if n_seen % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {n_seen}/{len(test_anchors_with_truth)} anchors, "
                  f"{elapsed:.1f}s "
                  f"({n_seen/elapsed:.1f} anchors/s)", file=sys.stderr)

    print(f"\nbuilt {len(out_rows):,} candidate rows", file=sys.stderr)
    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.out, compression="snappy", index=False)
    print(f"wrote -> {args.out}", file=sys.stderr)

    # ---- recall@K summary ---------------------------------------------------
    print("\nRecall@K (true-successor retrieved in top-K):", file=sys.stderr)
    for split in ("split_time", "split_event", "split_jurisdiction"):
        sub = out_df[out_df[split] == "test"]
        # For each anchor, did we retrieve ALL true successors?
        per_anc = sub.groupby("anchor_lei").apply(
            lambda g: (g["label"] == 1).any() and
                       (g[g["label"] == 1]["retrieval_rank"] > 0).all()
        )
        # Hits@K: at least one true positive in top-K
        per_anc_hit = sub.groupby("anchor_lei").apply(
            lambda g: ((g["label"] == 1) & (g["retrieval_rank"] > 0)).any()
        )
        print(f"  {split}: Hits@{args.k} = "
              f"{per_anc_hit.mean()*100:.1f}% ({per_anc_hit.sum()}/"
              f"{len(per_anc_hit)} anchors)", file=sys.stderr)


if __name__ == "__main__":
    main()
