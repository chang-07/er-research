"""Score a pairs file with a saved cross-encoder using the v6 prompt
(name + jurisdiction + form + city + year + parent + subsidiaries + ISO
country-name expansion + legal-jurisdiction subdivision + category +
hq country). Required for v8a/v8aL12/v10 fielded re-scoring.

Mirrors src/method/train_cross_encoder.py format_entity + text_for so the
prompt seen at inference matches the prompt seen during training.

Usage:
  python3 src/method/score_pairs_v6prompt.py \\
      --model-dir data/models/method_v8a_combined \\
      --pairs pairs_fielded.parquet \\
      --out-name method_v8a_combined_fielded
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse the exact v6 format_entity to avoid prompt drift
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_cross_encoder import format_entity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
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

    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])

    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code",
                  "legal_addr_city", "entity_creation_date",
                  "legal_jurisdiction", "category", "hq_addr_country"],
    ).to_pandas().drop_duplicates(subset=["lei"], keep="first")
    ent = ent[ent["lei"].isin(lei_set)].reset_index(drop=True)
    ent["creation_year"] = pd.to_datetime(ent["entity_creation_date"],
                                            utc=True, errors="coerce").dt.year
    print(f"  entities: {len(ent):,}", file=sys.stderr)

    print("loading relationships...", file=sys.stderr)
    rel = pq.read_table(
        os.path.join(args.data, "relationships.parquet"),
        columns=["start_lei", "end_lei", "relationship_type",
                 "relationship_status"],
    ).to_pandas()
    rel = rel[(rel["relationship_status"] == "ACTIVE")
              & (rel["relationship_type"].isin(
                  ["IS_DIRECTLY_CONSOLIDATED_BY",
                   "IS_ULTIMATELY_CONSOLIDATED_BY"]))]
    parent_of = dict(zip(rel["start_lei"], rel["end_lei"]))
    children_of = defaultdict(list)
    for child, parent in zip(rel["start_lei"], rel["end_lei"]):
        if len(children_of[parent]) < 5:
            children_of[parent].append(child)
    # name lookup needs to cover ALL leis, not just the lei_set subset,
    # because parents/children may be outside lei_set
    full_names = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name"],
    ).to_pandas().drop_duplicates("lei", keep="first")
    name_of = dict(zip(full_names["lei"], full_names["legal_name"].fillna("")))

    ent_map = ent.set_index("lei").to_dict(orient="index")

    def text_for(lei):
        r = ent_map.get(lei)
        if not r:
            return "(unknown)"
        parent_lei = parent_of.get(lei)
        parent_name = name_of.get(parent_lei) if parent_lei else None
        kids = children_of.get(lei, [])
        child_names = [name_of.get(c) for c in kids if name_of.get(c)]
        year = r.get("creation_year")
        year_str = None
        try:
            if year and not (isinstance(year, float) and (year != year)):
                year_str = str(int(year))
        except Exception:
            pass
        return format_entity(
            r.get("legal_name"), r.get("legal_addr_country"),
            r.get("legal_form_code"), city=r.get("legal_addr_city"),
            year=year_str, parent_name=parent_name, child_names=child_names,
            jurisdiction=r.get("legal_jurisdiction"),
            category=r.get("category"),
            hq_country=r.get("hq_addr_country"),
        )

    print(f"loading cross-encoder from {args.model_dir}...", file=sys.stderr)
    model = CrossEncoder(args.model_dir, device=device)

    print(f"scoring {len(pairs):,} pairs (v6 prompt)...", file=sys.stderr)
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
