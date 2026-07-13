"""Measure inference latency and report cost per method.

Outputs data/cost_latency.csv with columns:
  method, params (M), inference_device, ms_per_pair, dollars_per_1k_pairs

For LLM rows, dollars-per-1k is computed from per-call token estimates
(prompt-cached system + per-call user + ~17 output tokens) at published
list prices. Latency for LLMs is omitted (depends on network, batching).
"""
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
}

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)


def time_cheap(pairs, ent, n=1000):
    """Time exact_name, token_set, tfidf_char on n pairs each."""
    sub = pairs.head(n).copy()
    nmap = {l: normalize_name(n_) for l, n_ in zip(ent["lei"], ent["legal_name"])}
    tmap = {l: frozenset(n.split()) for l, n in nmap.items()}

    out = {}

    # exact_name
    t0 = time.perf_counter()
    a = sub["anchor_lei"].map(nmap).fillna("").values
    c = sub["candidate_lei"].map(nmap).fillna("").values
    _ = (a == c) & (a != "")
    out["exact_name"] = (time.perf_counter() - t0) * 1000 / n

    # token_set jaccard
    t0 = time.perf_counter()
    scores = np.zeros(n)
    for i, (la, lc) in enumerate(zip(sub["anchor_lei"], sub["candidate_lei"])):
        ta = tmap.get(la); tc = tmap.get(lc)
        if not ta or not tc:
            continue
        u = ta | tc
        scores[i] = len(ta & tc) / len(u) if u else 0.0
    out["token_set"] = (time.perf_counter() - t0) * 1000 / n

    # tfidf_char
    names = list({nmap[l] for l in ent["lei"] if l in nmap})[:5000]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                           max_features=20_000)
    mat = vec.fit_transform(names)
    lei2idx = {l: i for i, l in enumerate(ent["lei"][:5000]) if l in nmap}
    t0 = time.perf_counter()
    for la, lc in zip(sub["anchor_lei"], sub["candidate_lei"]):
        if la not in lei2idx or lc not in lei2idx:
            continue
        A = mat[lei2idx[la]]; C = mat[lei2idx[lc]]
        _ = A.multiply(C).sum()
    out["tfidf_char"] = (time.perf_counter() - t0) * 1000 / n
    return out


def llm_cost_per_1k(input_per_call=255, cached_input=165, user=85, output=60):
    """List-price dollar cost per 1K pair classifications.
    Models priced as of late 2025/early 2026.
    """
    # (model, $/MTok input, $/MTok output, $/MTok cached read)
    models = [
        ("claude-haiku-4-5",     1.00, 5.00, 0.10),
        ("claude-sonnet-4-6",    3.00, 15.0, 0.30),
        ("claude-opus-4-7",      5.0,  25.0, 0.50),
        ("gpt-4o-mini",          0.15, 0.60, None),
        ("gpt-4o",               2.50, 10.0, None),
        ("gemini-2.0-flash",     0.10, 0.40, None),
        ("gemini-2.5-flash",     0.30, 2.50, None),
        ("llama-3.1-8b-tg",      0.10, 0.10, None),  # Together AI
    ]
    rows = []
    for name, p_in, p_out, p_cache in models:
        if p_cache is not None:
            # No cache discount despite cache_control in the runner: the
            # ~165-token system prompt is below every current model's
            # minimum cacheable prefix (2048-4096 tokens), so the prefix
            # bills at full input price. Confirmed on the July 2026 Haiku
            # full run.
            in_cost = ((cached_input + user) * p_in * 1000) / 1_000_000
        else:
            in_cost = input_per_call * p_in * 1000 / 1_000_000
        out_cost = output * p_out * 1000 / 1_000_000
        rows.append({
            "method": f"llm_{name}",
            "params_M": "frontier" if "opus" in name or "4o" in name or "pro" in name else "small",
            "inference_device": "API",
            "ms_per_pair": float("nan"),
            "dollars_per_1k_pairs": round(in_cost + out_cost, 4),
        })
    return rows


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet"),
                           columns=["anchor_lei", "candidate_lei"]).to_pandas()
    ent = pq.read_table(os.path.join(data, "entities.parquet"),
                         columns=["lei", "legal_name"]).to_pandas().drop_duplicates("lei")

    print("Timing cheap baselines on 1000 pairs (CPU, single-thread)...",
          file=sys.stderr)
    cheap = time_cheap(pairs, ent, n=1000)
    rows = [
        {"method": "exact_name", "params_M": 0,    "inference_device": "CPU",
         "ms_per_pair": round(cheap["exact_name"], 4),
         "dollars_per_1k_pairs": 0.0},
        {"method": "token_set",  "params_M": 0,    "inference_device": "CPU",
         "ms_per_pair": round(cheap["token_set"], 4),
         "dollars_per_1k_pairs": 0.0},
        {"method": "tfidf_char", "params_M": 0,    "inference_device": "CPU",
         "ms_per_pair": round(cheap["tfidf_char"], 4),
         "dollars_per_1k_pairs": 0.0},
        # bi-encoder + cross-encoder use prior measured wall times:
        # E5-small encode + cosine: 40s for 125K pairs => ~0.32 ms/pair on MPS
        # ms-marco MiniLM-L-6 cross-encoder: 79.6s for 125K pairs => ~0.64 ms/pair
        # All MPS, M-series. Marginal cost = $0 (laptop).
        {"method": "sentence_encoder (E5-small, zero-shot)", "params_M": 33,
         "inference_device": "Apple MPS", "ms_per_pair": 0.32,
         "dollars_per_1k_pairs": 0.0},
        {"method": "method_v2_full (E5-small fine-tune)", "params_M": 33,
         "inference_device": "Apple MPS", "ms_per_pair": 0.32,
         "dollars_per_1k_pairs": 0.0},
        {"method": "method_v2_crossenc (MiniLM-L-6 fine-tune)", "params_M": 22,
         "inference_device": "Apple MPS", "ms_per_pair": 0.64,
         "dollars_per_1k_pairs": 0.0},
    ]
    rows.extend(llm_cost_per_1k())

    df = pd.DataFrame(rows)
    out_path = os.path.join(data, "cost_latency.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote -> {out_path}")
    print(f"\nNote: LLM rows omit ms_per_pair (depends on network/batching).")
    print(f"Non-LLM marginal $/1k is 0 (single-laptop inference).")


if __name__ == "__main__":
    main()
