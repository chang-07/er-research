"""Score an external (text_a, text_b, label) test set with zero-shot E5-small-v2.

Baseline comparison for the cross-dataset transfer experiment: if our
trained cross-encoder beats zero-shot E5 on Walmart-Amazon by a meaningful
margin, that's evidence the lifecycle-ER training transfers to point-in-
time product matching. If they're tied, our training is GLEIF-specific.

Input format: same as score_external.py (Ditto COL/VAL markup, tab-separated).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def strip_ditto_markup(s: str) -> str:
    parts = re.split(r"COL\s+\S+\s+VAL\s+", s)
    parts = [p.strip() for p in parts[1:] if p.strip()]
    return " | ".join(parts)


def load_pairs(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            ta, tb, label = cols
            rows.append({
                "id": i,
                "text_a": strip_ditto_markup(ta),
                "text_b": strip_ditto_markup(tb),
                "label": int(label),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/e5-small-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model} on {device}...", file=sys.stderr)
    model = SentenceTransformer(args.model, device=device)

    test = load_pairs(args.test_file)
    print(f"  test pairs: {len(test):,}, positives: {(test.label==1).sum():,}",
          file=sys.stderr)

    texts_a = ["query: " + t for t in test["text_a"]]
    texts_b = ["query: " + t for t in test["text_b"]]
    emb_a = model.encode(texts_a, batch_size=128, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True)
    emb_b = model.encode(texts_b, batch_size=128, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True)
    scores = (emb_a * emb_b).sum(axis=1)
    # cosine in [-1, 1]; clip to [0, 1] for prob-like interpretation
    scores = np.clip(scores, 0, 1)

    test["score"] = scores

    thr = 0.5
    if args.train_file:
        train = load_pairs(args.train_file)
        ta = ["query: " + t for t in train["text_a"]]
        tb = ["query: " + t for t in train["text_b"]]
        ea = model.encode(ta, batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
        eb = model.encode(tb, batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
        tr_scores = np.clip((ea * eb).sum(axis=1), 0, 1)
        from sklearn.metrics import precision_recall_curve
        if train["label"].nunique() >= 2:
            p, r, t = precision_recall_curve(train["label"], tr_scores)
            p, r = p[:-1], r[:-1]
            f1 = 2 * p * r / np.where(p + r > 0, p + r, 1)
            thr = float(t[int(np.nanargmax(f1))])
        print(f"  tuned threshold (train F1-opt) = {thr:.3f}", file=sys.stderr)

    pred = (test["score"] >= thr).astype(int)
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                   roc_auc_score, average_precision_score)
    p = precision_score(test["label"], pred, zero_division=0)
    r = recall_score(test["label"], pred, zero_division=0)
    f1 = f1_score(test["label"], pred, zero_division=0)
    auroc = roc_auc_score(test["label"], test["score"])
    auprc = average_precision_score(test["label"], test["score"])

    test.to_parquet(args.out, compression="snappy", index=False)
    print()
    print(f"=== external transfer (E5 zero-shot) result ===")
    print(f"test: {os.path.basename(args.test_file)}  n={len(test)}  positives={(test.label==1).sum()}")
    print(f"AUROC = {auroc:.3f}  AUPRC = {auprc:.3f}")
    print(f"F1@tau={thr:.3f}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
