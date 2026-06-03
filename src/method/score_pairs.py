"""Score a pairs.parquet-style file with a saved sentence-transformers
model (either cross-encoder or bi-encoder). For the fielded-retriever
experiment: we score pairs_fielded.parquet with the saved cross-encoder
to reproduce a real two-stage retrieve-then-rerank pipeline.

Usage:
  python3 src/method/score_pairs.py \\
      --model-dir data/models/method_v2_crossenc \\
      --pairs pairs_fielded.parquet \\
      --out-name method_v2_crossenc_on_fielded
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def format_entity_basic(name, country, form):
    name = (name or "").strip() or "(unknown)"
    country = (country or "??").strip() or "??"
    form = (form or "??").strip() or "??"
    return f"{name} | jurisdiction: {country} | form: {form}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pairs", default="pairs_fielded.parquet")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cross", action="store_true",
                    help="treat the model as a cross-encoder (default)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    pairs = pq.read_table(
        os.path.join(args.data, args.pairs),
        columns=["anchor_lei", "candidate_lei"],
    ).to_pandas()
    print(f"loaded {len(pairs):,} pairs", file=sys.stderr)

    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas().drop_duplicates("lei", keep="first")
    ent_map = ent.set_index("lei").to_dict(orient="index")

    def text_for(lei):
        r = ent_map.get(lei)
        if not r:
            return "(unknown)"
        return format_entity_basic(r.get("legal_name"),
                                     r.get("legal_addr_country"),
                                     r.get("legal_form_code"))

    print(f"loading cross-encoder from {args.model_dir}...", file=sys.stderr)
    model = CrossEncoder(args.model_dir, device=device)

    print(f"scoring {len(pairs):,} pairs...", file=sys.stderr)
    t0 = time.time()
    inp = [(text_for(a), text_for(c))
            for a, c in zip(pairs["anchor_lei"], pairs["candidate_lei"])]
    scores = model.predict(inp, batch_size=args.batch_size,
                            show_progress_bar=True, convert_to_numpy=True)
    scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid
    print(f"  scored in {time.time()-t0:.0f}s "
          f"(mean={float(scores.mean()):.3f})", file=sys.stderr)

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out_name}.parquet")
    pd.DataFrame({
        "anchor_lei": pairs["anchor_lei"],
        "candidate_lei": pairs["candidate_lei"],
        "score": scores.astype(float),
    }).to_parquet(out_path, compression="snappy", index=False)
    print(f"wrote -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
