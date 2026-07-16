"""One-shot checks for the methodology-gap fixes (June 2026).

1. Anti-similarity baselines: every similarity baseline negated
   (rank by 1 - similarity), evaluated with the exact evaluate.py
   protocol. Quantifies the "pick the most name-dissimilar candidate"
   exploit that the benchmark's construction invites.
2. Chance level: expected random-ranking Hits@1 per split.
3. Train/test entity overlap: fraction of test-anchor candidate (and
   gold-successor) LEIs that also appear in that regime's train pairs.
4. Cross-jurisdiction positives: how many true successors sit in a
   different country than the anchor (= recall ceiling imposed by the
   fielded simulation's same-country filter).
5. Date hygiene: NaT / pre-1900 anchor_event_date rows and how
   split_time assigns them.
6. Temperature scaling: ECE/Brier for the headline cross-encoder
   before/after a single temperature fitted on each regime's train
   subset.

Read-only with respect to benchmark artifacts; prints a report.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
SPLITS = ("split_time", "split_event", "split_jurisdiction")


def _expected_arrays(df_test):
    from ranking import per_anchor_expected
    eh1, errs = [], []
    for _a, hks, err in per_anchor_expected(df_test, ks=(1,)):
        eh1.append(hks[1])
        errs.append(err)
    return np.asarray(eh1), np.asarray(errs)


def retrieval(df_test):
    eh1, errs = _expected_arrays(df_test)
    if len(eh1) == 0:
        return float("nan"), float("nan"), 0
    return float(eh1.mean()), float(errs.mean()), len(eh1)


def boot_ci(df_test, n_boots=200, seed=42):
    eh1, _ = _expected_arrays(df_test)
    if len(eh1) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [eh1[rng.integers(0, len(eh1), len(eh1))].mean()
             for _ in range(n_boots)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ece10(y, p):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


pairs = pq.read_table(os.path.join(DATA, "pairs.parquet")).to_pandas()

print("=" * 72)
print("1. ANTI-SIMILARITY BASELINES (rank by negated similarity score)")
print("=" * 72)
for base in ("token_set", "tfidf_char", "sentence_encoder", "exact_name"):
    preds = pq.read_table(
        os.path.join(DATA, "predictions", f"{base}.parquet"),
        columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
    df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
    df["score"] = -df["score"].fillna(0.0)
    for regime in SPLITS:
        test = df[df[regime] == "test"]
        auroc = roc_auc_score(test["label"], test["score"])
        h1, mrr, n = retrieval(test)
        lo, hi = boot_ci(test)
        print(f"anti-{base:17s} {regime:20s} AUROC={auroc:.3f} "
              f"H@1={h1:.3f} [{lo:.3f},{hi:.3f}] MRR={mrr:.3f} n={n}")

print()
print("=" * 72)
print("2. CHANCE LEVEL (random ranking: per-anchor n_pos/n_candidates)")
print("=" * 72)
for regime in SPLITS:
    test = pairs[pairs[regime] == "test"]
    exp, n = [], 0
    for _, sub in test.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        exp.append(sub["label"].sum() / len(sub))
        n += 1
    print(f"{regime:20s} expected random Hits@1 = {np.mean(exp):.3f} "
          f"(n={n} anchors, mean pool size "
          f"{np.mean([1/e for e in exp]) if exp else float('nan'):.1f})")

print()
print("=" * 72)
print("3. TRAIN/TEST ENTITY OVERLAP (per regime)")
print("=" * 72)
for regime in SPLITS:
    tr = pairs[pairs[regime] == "train"]
    te = pairs[pairs[regime] == "test"]
    train_ents = set(tr["anchor_lei"]) | set(tr["candidate_lei"])
    te_cand = te["candidate_lei"]
    te_gold = te.loc[te["label"] == 1, "candidate_lei"]
    te_anch = te["anchor_lei"].drop_duplicates()
    print(f"{regime:20s} test candidates seen in train: "
          f"{te_cand.isin(train_ents).mean():.1%}  "
          f"gold successors seen in train: {te_gold.isin(train_ents).mean():.1%}  "
          f"test anchors seen in train: {te_anch.isin(train_ents).mean():.1%}")

print()
print("=" * 72)
print("4. CROSS-JURISDICTION POSITIVES (same-country-filter recall ceiling)")
print("=" * 72)
ent = pq.read_table(os.path.join(DATA, "entities.parquet"),
                    columns=["lei", "legal_addr_country"]).to_pandas()
cmap = dict(zip(ent["lei"], ent["legal_addr_country"]))
pos = pairs[pairs["label"] == 1].copy()
pos["succ_country"] = pos["candidate_lei"].map(cmap)
pos["xjur"] = (pos["succ_country"].notna()
               & (pos["succ_country"] != pos["anchor_country"]))
print(f"all positives: {pos['xjur'].mean():.2%} cross-jurisdiction "
      f"({pos['xjur'].sum():,}/{len(pos):,})")
for regime in SPLITS:
    sub = pos[pos[regime] == "test"]
    print(f"{regime:20s} test positives cross-jurisdiction: "
          f"{sub['xjur'].mean():.2%} ({sub['xjur'].sum()}/{len(sub)})")

print()
print("=" * 72)
print("5. DATE HYGIENE (anchor_event_date in released pairs)")
print("=" * 72)
d = pd.to_datetime(pairs["anchor_event_date"], errors="coerce", utc=True)
nat = d.isna()
garbage = d.dt.year < 1900
print(f"NaT anchor_event_date rows: {nat.sum():,} ({nat.mean():.1%})")
print(f"pre-1900 (garbage) dates:   {garbage.sum():,}")
print("split_time assignment for NaT rows:")
print(pairs.loc[nat, "split_time"].value_counts().to_string())
print("split_time assignment for pre-1900 rows:")
print(pairs.loc[garbage.fillna(False), "split_time"].value_counts().to_string())

print()
print("=" * 72)
print("6. TEMPERATURE SCALING (headline L-6 cross-encoder)")
print("=" * 72)
preds = pq.read_table(
    os.path.join(DATA, "predictions", "method_v2_crossenc.parquet"),
    columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
df["score"] = df["score"].fillna(0.0).clip(1e-6, 1 - 1e-6)
print(f"score range: [{df['score'].min():.4f}, {df['score'].max():.4f}] "
      "(treating as probabilities)")
for regime in SPLITS:
    tr = df[df[regime] == "train"]
    te = df[df[regime] == "test"]
    logits_tr = np.log(tr["score"] / (1 - tr["score"]))
    logits_te = np.log(te["score"] / (1 - te["score"]))
    y_tr, y_te = tr["label"].to_numpy(), te["label"].to_numpy()
    # 1-D grid search for T minimizing train NLL
    Ts = np.linspace(0.3, 8.0, 78)
    best_T, best_nll = 1.0, np.inf
    for T in Ts:
        p = 1 / (1 + np.exp(-logits_tr / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)).mean()
        if nll < best_nll:
            best_nll, best_T = nll, T
    p_raw = te["score"].to_numpy()
    p_cal = 1 / (1 + np.exp(-logits_te / best_T))
    print(f"{regime:20s} T={best_T:.2f}  "
          f"ECE {ece10(y_te, p_raw):.3f} -> {ece10(y_te, p_cal):.3f}   "
          f"Brier {np.mean((p_raw-y_te)**2):.3f} -> {np.mean((p_cal-y_te)**2):.3f}")
print()
print("done.")
