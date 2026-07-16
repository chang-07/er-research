"""Part 2 of gap checks: where does anti-similarity fail?

1. anti-token_set on the TF-IDF pool (pairs_v3) — does the trick
   transfer across miners?
2. anti-token_set on the FIELDED pool (pairs_fielded) — conditional
   Hits@1 vs the cross-encoder's 46-58%.
3. Per-event-type anti-token_set Hits@1 on the TIME split — renames
   should collapse.
4. Memorization check: headline L-6 Hits@1 split by whether the gold
   successor entity was seen anywhere in that regime's train pairs.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
SPLITS = ("split_time", "split_event", "split_jurisdiction")


def h1_mrr(df_test):
    from ranking import per_anchor_expected
    eh1, errs = [], []
    for _a, hks, err in per_anchor_expected(df_test, ks=(1,)):
        eh1.append(hks[1])
        errs.append(err)
    if not eh1:
        return float("nan"), float("nan"), 0
    return float(np.mean(eh1)), float(np.mean(errs)), len(eh1)


print("=" * 72)
print("A. ANTI-TOKEN_SET ON THE TF-IDF POOL (pairs_v3)")
print("=" * 72)
pairs3 = pq.read_table(os.path.join(DATA, "pairs_v3.parquet")).to_pandas()
p3 = pq.read_table(os.path.join(DATA, "predictions", "token_set_v3.parquet"),
                   columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
df = pairs3.merge(p3, on=["anchor_lei", "candidate_lei"], how="left")
df["score"] = -df["score"].fillna(0.0)
for regime in SPLITS:
    test = df[df[regime] == "test"]
    auroc = roc_auc_score(test["label"], test["score"])
    h1, mrr, n = h1_mrr(test)
    print(f"{regime:20s} AUROC={auroc:.3f} H@1={h1:.3f} MRR={mrr:.3f} n={n}")

print()
print("=" * 72)
print("B. FIELDED POOL: anti-token_set vs cross-encoder (conditional H@1)")
print("=" * 72)
pf = pq.read_table(os.path.join(DATA, "pairs_fielded.parquet")).to_pandas()
print("fielded pairs columns:", list(pf.columns))
for name, predfile, neg in [
    ("anti-token_set", "token_set_fielded.parquet", True),
    ("token_set", "token_set_fielded.parquet", False),
    ("cross-enc L-6", "method_v2_crossenc_fielded.parquet", False),
]:
    pr = pq.read_table(os.path.join(DATA, "predictions", predfile),
                       columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
    d = pf.merge(pr, on=["anchor_lei", "candidate_lei"], how="left")
    d["score"] = d["score"].fillna(0.0) * (-1 if neg else 1)
    row = []
    for regime in SPLITS:
        col = regime if regime in d.columns else None
        if col is None:
            continue
        test = d[d[col] == "test"]
        h1, mrr, n = h1_mrr(test)
        row.append(f"{regime.split('_')[1]}: H@1={h1:.3f} (n={n})")
    print(f"{name:16s} " + "  ".join(row))

print()
print("=" * 72)
print("C. PER-EVENT-TYPE anti-token_set H@1 ON TIME SPLIT")
print("=" * 72)
pairs = pq.read_table(os.path.join(DATA, "pairs.parquet")).to_pandas()
ts = pq.read_table(os.path.join(DATA, "predictions", "token_set.parquet"),
                   columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
df = pairs.merge(ts, on=["anchor_lei", "candidate_lei"], how="left")
df["score"] = -df["score"].fillna(0.0)
test = df[df["split_time"] == "test"]
for ev, sub in test.groupby("anchor_event_type"):
    h1, mrr, n = h1_mrr(sub)
    if n >= 4:
        print(f"{ev:30s} H@1={h1:.3f} (n={n})")

print()
print("=" * 72)
print("D. MEMORIZATION CHECK: L-6 H@1 by gold-successor-seen-in-train")
print("=" * 72)
ce = pq.read_table(os.path.join(DATA, "predictions", "method_v2_crossenc.parquet"),
                   columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
df = pairs.merge(ce, on=["anchor_lei", "candidate_lei"], how="left")
df["score"] = df["score"].fillna(0.0)
for regime in ("split_time", "split_event"):
    tr = pairs[pairs[regime] == "train"]
    train_ents = set(tr["anchor_lei"]) | set(tr["candidate_lei"])
    test = df[df[regime] == "test"]
    gold = test[test["label"] == 1].groupby("anchor_lei")["candidate_lei"].first()
    seen_anchors = set(gold[gold.isin(train_ents)].index)
    t_seen = test[test["anchor_lei"].isin(seen_anchors)]
    t_unseen = test[~test["anchor_lei"].isin(seen_anchors)]
    h1s, _, ns = h1_mrr(t_seen)
    h1u, _, nu = h1_mrr(t_unseen)
    print(f"{regime:20s} gold-seen: H@1={h1s:.3f} (n={ns})   "
          f"gold-unseen: H@1={h1u:.3f} (n={nu})")
print()
print("done.")
