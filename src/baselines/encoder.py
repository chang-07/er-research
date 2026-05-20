"""Sentence-encoder baseline: zero-shot cosine of legal-name embeddings.

Uses intfloat/e5-small-v2 by default — a 33M-param dual-encoder tuned for
retrieval, runs comfortably on CPU. Embeddings are L2-normalized and the
pair score is the dot product.

This is the "modern pretrained encoder, zero-shot" baseline from the plan.
Same input contract as cheap.py: read pairs.parquet, score every pair, write
data/predictions/sentence_encoder.parquet.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="intfloat/e5-small-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--name", default="sentence_encoder",
                    help="prediction file basename")
    args = ap.parse_args()

    # Import lazily so the script imports without torch installed
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("Loading pairs + entity names...", file=sys.stderr)
    pairs = pq.read_table(os.path.join(args.data, "pairs.parquet"),
                          columns=["anchor_lei", "candidate_lei"]).to_pandas()
    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])
    ent = pq.read_table(os.path.join(args.data, "entities.parquet"),
                        columns=["lei", "legal_name"]).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].reset_index(drop=True)
    ent["legal_name"] = ent["legal_name"].fillna("")
    print(f"  {len(pairs):,} pairs, {len(ent):,} unique LEIs to encode",
          file=sys.stderr)

    # E5 expects "query: ..." style prefixes for asymmetric retrieval.
    # For a symmetric name-match task, use "query: " on both sides; this
    # outperforms no-prefix on E5 family models per the E5 paper.
    use_prefix = "e5" in args.model.lower()
    prefix = "query: " if use_prefix else ""
    texts = [prefix + n for n in ent["legal_name"]]

    print(f"Encoding {len(texts):,} names "
          f"(batch={args.batch_size}, prefix={'yes' if use_prefix else 'no'})...",
          file=sys.stderr)
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  encoded in {time.time()-t0:.1f}s -> shape {embs.shape}",
          file=sys.stderr)

    lei_to_idx = {lei: i for i, lei in enumerate(ent["lei"])}
    a_idx = pairs["anchor_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    c_idx = pairs["candidate_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    valid = (a_idx >= 0) & (c_idx >= 0)

    scores = np.zeros(len(pairs), dtype=np.float32)
    if valid.any():
        A = embs[a_idx[valid]]
        C = embs[c_idx[valid]]
        # cosine of L2-normalized vectors = elementwise * + sum
        sim = (A * C).sum(axis=1)
        scores[valid] = sim
    print(f"  scored {len(pairs):,} pairs "
          f"(mean={scores.mean():.3f}, median={np.median(scores):.3f})",
          file=sys.stderr)

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame({"anchor_lei": pairs["anchor_lei"],
                        "candidate_lei": pairs["candidate_lei"],
                        "score": scores.astype(float)})
    out_path = os.path.join(out_dir, f"{args.name}.parquet")
    out.to_parquet(out_path, compression="snappy", index=False)
    print(f"wrote -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
