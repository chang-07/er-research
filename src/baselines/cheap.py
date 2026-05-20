"""Cheap CPU baselines on TemporalLink v1 pairs.

Three baselines, all <1 min on 125K pairs:

  exact_name   normalized legal name equality (0 or 1)
  token_set    Jaccard similarity over normalized-name token sets
  tfidf_char   L2-normalized char-3gram TF-IDF cosine, fit on the union of
               legal_name strings appearing in pairs.parquet

Each writes data/predictions/<baseline>.parquet with
  (anchor_lei, candidate_lei, score)
which the eval script joins back against pairs.parquet for labels and splits.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

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
    args = ap.parse_args()

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading pairs...", file=sys.stderr)
    pairs = pq.read_table(os.path.join(args.data, "pairs.parquet"),
                          columns=["anchor_lei", "candidate_lei"]).to_pandas()
    print(f"  {len(pairs):,} pairs", file=sys.stderr)

    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])
    print(f"  {len(lei_set):,} unique LEIs", file=sys.stderr)

    print("Loading entity names...", file=sys.stderr)
    ent = pq.read_table(os.path.join(args.data, "entities.parquet"),
                        columns=["lei", "legal_name"]).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].reset_index(drop=True)
    ent["nname"] = ent["legal_name"].apply(normalize_name)
    ent["tokens"] = ent["nname"].apply(lambda s: frozenset(s.split()))
    nname_map = dict(zip(ent["lei"], ent["nname"]))
    tok_map = dict(zip(ent["lei"], ent["tokens"]))
    legal_map = dict(zip(ent["lei"], ent["legal_name"].fillna("")))
    print(f"  {len(ent):,} entities loaded", file=sys.stderr)

    # ---- exact_name --------------------------------------------------------
    t0 = time.time()
    print("\nexact_name...", file=sys.stderr)
    a_n = pairs["anchor_lei"].map(nname_map).fillna("").values
    c_n = pairs["candidate_lei"].map(nname_map).fillna("").values
    scores = np.where((a_n != "") & (a_n == c_n), 1.0, 0.0)
    out = pd.DataFrame({"anchor_lei": pairs["anchor_lei"],
                        "candidate_lei": pairs["candidate_lei"],
                        "score": scores})
    out.to_parquet(os.path.join(out_dir, "exact_name.parquet"),
                   compression="snappy", index=False)
    print(f"  scored {len(out):,} pairs in {time.time()-t0:.1f}s "
          f"(positive predictions: {int(scores.sum()):,})", file=sys.stderr)

    # ---- token_set (Jaccard) ----------------------------------------------
    t0 = time.time()
    print("\ntoken_set...", file=sys.stderr)
    scores = np.zeros(len(pairs))
    for i, (a, c) in enumerate(zip(pairs["anchor_lei"], pairs["candidate_lei"])):
        ta = tok_map.get(a)
        tc = tok_map.get(c)
        if not ta or not tc:
            continue
        u = ta | tc
        scores[i] = len(ta & tc) / len(u) if u else 0.0
    out = pd.DataFrame({"anchor_lei": pairs["anchor_lei"],
                        "candidate_lei": pairs["candidate_lei"],
                        "score": scores})
    out.to_parquet(os.path.join(out_dir, "token_set.parquet"),
                   compression="snappy", index=False)
    print(f"  scored {len(out):,} pairs in {time.time()-t0:.1f}s "
          f"(mean={scores.mean():.3f}, median={np.median(scores):.3f})", file=sys.stderr)

    # ---- tfidf_char --------------------------------------------------------
    t0 = time.time()
    print("\ntfidf_char...", file=sys.stderr)
    leis = list(ent["lei"])
    names = [legal_map[lei] or "" for lei in leis]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                          min_df=2, max_features=60_000, dtype=np.float32)
    mat = vec.fit_transform(names)
    print(f"  TF-IDF matrix: {mat.shape}, nnz={mat.nnz:,}", file=sys.stderr)
    lei_to_idx = {lei: i for i, lei in enumerate(leis)}

    a_idx = pairs["anchor_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    c_idx = pairs["candidate_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    valid = (a_idx >= 0) & (c_idx >= 0)
    scores = np.zeros(len(pairs), dtype=np.float32)
    if valid.any():
        A = mat[a_idx[valid]]  # sparse rows
        C = mat[c_idx[valid]]
        # cosine of L2-normalized = element-wise multiply, sum across cols
        prod = A.multiply(C)
        dots = np.asarray(prod.sum(axis=1)).flatten()
        scores[valid] = dots
    out = pd.DataFrame({"anchor_lei": pairs["anchor_lei"],
                        "candidate_lei": pairs["candidate_lei"],
                        "score": scores.astype(float)})
    out.to_parquet(os.path.join(out_dir, "tfidf_char.parquet"),
                   compression="snappy", index=False)
    print(f"  scored {len(out):,} pairs in {time.time()-t0:.1f}s "
          f"(mean={scores.mean():.3f}, median={np.median(scores):.3f})", file=sys.stderr)

    print(f"\nWrote predictions to {out_dir}/", file=sys.stderr)

if __name__ == "__main__":
    main()
