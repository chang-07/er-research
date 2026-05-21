"""Re-score pairs.parquet (or any pairs file) with a saved cross-encoder
using symmetric averaging: for each (anchor, candidate), score both
(anchor, candidate) and (candidate, anchor) and average the two sigmoid
outputs.

Cross-encoders are asymmetric by design (the joint encoding processes the
two inputs in order), but lifecycle ER is direction-agnostic — we don't
care whether the predecessor came first or the successor did when
deciding "are these the same continuing entity." Symmetric scoring
typically adds +1 to +2 pp Hits@1 at essentially zero cost (2x inference
time on tiny model).

Usage:
  python3 src/method/score_symmetric.py \\
      --model-dir data/models/method_v4_crossenc_enriched \\
      --pairs pairs.parquet \\
      --out-name method_v4_crossenc_enriched_sym
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse the enriched format_entity from train_cross_encoder so symmetric
# scoring sees the same prompt as training.
sys.path.insert(0, os.path.dirname(__file__))
from train_cross_encoder import format_entity


def build_text_for(data):
    """Returns a closure text_for(lei) that uses enriched prompts."""
    from collections import defaultdict

    ent = pq.read_table(
        os.path.join(data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code",
                  "legal_addr_city", "entity_creation_date"],
    ).to_pandas().drop_duplicates(subset=["lei"], keep="first")
    ent["creation_year"] = pd.to_datetime(ent["entity_creation_date"],
                                            utc=True, errors="coerce").dt.year

    rel = pq.read_table(
        os.path.join(data, "relationships.parquet"),
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
    name_of = dict(zip(ent["lei"], ent["legal_name"].fillna("")))
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
        )
    return text_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pairs", default="pairs.parquet")
    ap.add_argument("--out-name", required=True)
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

    text_for = build_text_for(args.data)
    print(f"loading {args.model_dir}...", file=sys.stderr)
    model = CrossEncoder(args.model_dir, device=device)

    # Forward pass: (anchor, candidate)
    print("forward pass (a, c)...", file=sys.stderr)
    inp_fwd = [(text_for(a), text_for(c))
               for a, c in zip(pairs["anchor_lei"], pairs["candidate_lei"])]
    s_fwd = model.predict(inp_fwd, batch_size=128, show_progress_bar=True,
                           convert_to_numpy=True)
    s_fwd = 1.0 / (1.0 + np.exp(-s_fwd))

    # Reverse pass: (candidate, anchor)
    print("reverse pass (c, a)...", file=sys.stderr)
    inp_rev = [(text_for(c), text_for(a))
               for a, c in zip(pairs["anchor_lei"], pairs["candidate_lei"])]
    s_rev = model.predict(inp_rev, batch_size=128, show_progress_bar=True,
                           convert_to_numpy=True)
    s_rev = 1.0 / (1.0 + np.exp(-s_rev))

    scores = (s_fwd + s_rev) / 2.0
    print(f"  scored (mean fwd={s_fwd.mean():.3f}, rev={s_rev.mean():.3f}, "
          f"sym={scores.mean():.3f})", file=sys.stderr)

    preds_dir = os.path.join(args.data, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"{args.out_name}.parquet")
    pd.DataFrame({
        "anchor_lei": pairs["anchor_lei"],
        "candidate_lei": pairs["candidate_lei"],
        "score": scores.astype(float),
    }).to_parquet(out_path, compression="snappy", index=False)
    print(f"wrote -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
