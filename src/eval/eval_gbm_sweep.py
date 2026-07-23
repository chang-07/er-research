"""Score the GBM confound-sweep variants against the cross-encoder.

The adversarial review's central objection: every off-pool GBM-vs-CE
comparison confounds architecture with (a) negative source and (b)
feature family. This scores the four de-confounded variants on all
three pools, plus the fielded pool's strict retrieval-conditional and
end-to-end views, using the same expectation-based tie handling as
every other exhibit.

Usage: python3 src/eval/eval_gbm_sweep.py [data]
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from ranking import per_anchor_expected  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
SPLITS = ("split_time", "split_event", "split_jurisdiction")

VARIANTS = [
    ("GBM v1negs/name  (paper)", "features_gbm"),
    ("GBM v3negs/name  (paper)", "features_gbm_trnv3"),
    ("A: E5negs/name",           "features_gbm_trnv2"),
    ("B: v1negs/full",           "features_gbm_fullfeat"),
    ("C: E5negs/full",           "features_gbm_trnv2full"),
    ("D: v3negs/full",           "features_gbm_trnv3full"),
    ("Cross-encoder L-6",        None),  # special-cased file map
]
CE_FILES = {"bench": "method_v2_crossenc.parquet",
            "v3": "method_v2_crossenc_repro_on_pairs_v3.parquet",
            "fld": "method_v2_crossenc_fielded.parquet"}

POOLS = {"bench": "pairs.parquet", "v3": "pairs_v3.parquet",
         "fld": "pairs_fielded.parquet"}
pools = {k: pq.read_table(os.path.join(DATA, f)).to_pandas()
         for k, f in POOLS.items()}


def merged(pool_key, predfile):
    pr = pq.read_table(os.path.join(DATA, "predictions", predfile),
                       columns=["anchor_lei", "candidate_lei", "score"]
                       ).to_pandas()
    d = pools[pool_key].merge(pr, on=["anchor_lei", "candidate_lei"],
                              how="left")
    d["score"] = d["score"].fillna(0.0)
    return d


def eh1(df, regime):
    t = df[df[regime] == "test"]
    vals = [h[1] for _a, h, _e in per_anchor_expected(t, ks=(1,))]
    return float(np.mean(vals)) if vals else float("nan")


def fielded_views(predfile):
    """(gold-included, retrieval-conditional, end-to-end) per split."""
    d = merged("fld", predfile)
    out = {}
    for reg in SPLITS:
        t = d[d[reg] == "test"]
        gold_ret = set(t[(t.label == 1)
                         & (t.retrieval_rank >= 1)]["anchor_lei"])
        n_all = t[t.label == 1]["anchor_lei"].nunique()
        incl = [h[1] for _a, h, _e in per_anchor_expected(t, ks=(1,))]
        cond_rows = t[t["anchor_lei"].isin(gold_ret)]
        cond = [h[1] for _a, h, _e in per_anchor_expected(cond_rows, ks=(1,))]
        out[reg] = (float(np.mean(incl)) if incl else float("nan"),
                    float(np.mean(cond)) if cond else float("nan"),
                    float(np.sum(cond)) / n_all if n_all else float("nan"))
    return out


hdr = (f"{'variant':26s} | bench T/E/J        | v3 T/E/J           "
       f"| fld incl T/E/J     | fld cond T/E/J     | fld e2e T/E/J")
print(hdr)
print("-" * len(hdr))
for name, stem in VARIANTS:
    files = CE_FILES if stem is None else {
        "bench": f"{stem}.parquet", "v3": f"{stem}_v3.parquet",
        "fld": f"{stem}_fielded.parquet"}
    if not all(os.path.exists(os.path.join(DATA, "predictions", f))
               for f in files.values()):
        print(f"{name:26s} | (predictions missing)")
        continue
    b = [eh1(merged("bench", files["bench"]), r) for r in SPLITS]
    v = [eh1(merged("v3", files["v3"]), r) for r in SPLITS]
    fv = fielded_views(files["fld"])
    incl = [fv[r][0] for r in SPLITS]
    cond = [fv[r][1] for r in SPLITS]
    e2e = [fv[r][2] for r in SPLITS]
    fmt = lambda xs: "/".join(f"{x:.3f}" for x in xs)
    print(f"{name:26s} | {fmt(b):18s} | {fmt(v):18s} "
          f"| {fmt(incl):18s} | {fmt(cond):18s} | {fmt(e2e)}")
