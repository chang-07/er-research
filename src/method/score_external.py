"""Score an external (text_a, text_b, label) test set with a saved
sentence-transformers CrossEncoder checkpoint.

Used for the cross-dataset transfer experiment: trains-on-GLEIF cross-encoder
applied zero-shot to Magellan Walmart-Amazon to see whether the lifecycle-ER
framework generalizes beyond financial entities.

Input format: tab-separated lines of `text_a \t text_b \t label`,
where text_a/text_b may contain Ditto's `COL <name> VAL <value>` markup
(we strip the markup and keep the values for a vanilla cross-encoder input).

Output: data/predictions/<out-name>.parquet  with (id, text_a, text_b, label, score),
plus prints F1 at best train-tuned threshold (uniform threshold = 0.5 if no
train file supplied).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def strip_ditto_markup(s: str) -> str:
    # Replace "COL <name> VAL " with just the value, separated by " | "
    parts = re.split(r"COL\s+\S+\s+VAL\s+", s)
    # parts[0] is the prefix before the first COL (usually empty or whitespace)
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
    ap.add_argument("--model-dir", required=True,
                    help="path to saved CrossEncoder checkpoint")
    ap.add_argument("--test-file", required=True,
                    help="tab-separated text_a\\ttext_b\\tlabel file")
    ap.add_argument("--train-file", default=None,
                    help="optional train file to tune threshold")
    ap.add_argument("--out", required=True,
                    help="output parquet path")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    print(f"loading {args.model_dir}...", file=sys.stderr)
    model = CrossEncoder(args.model_dir, device=device)

    test = load_pairs(args.test_file)
    print(f"  test pairs: {len(test):,}, positives: {(test.label==1).sum():,}",
          file=sys.stderr)

    inp = [(r["text_a"], r["text_b"]) for _, r in test.iterrows()]
    scores = model.predict(inp, batch_size=128, show_progress_bar=True,
                            convert_to_numpy=True)
    scores = 1.0 / (1.0 + np.exp(-scores))

    test["score"] = scores

    # Threshold tuning (if train file provided)
    thr = 0.5
    if args.train_file:
        train = load_pairs(args.train_file)
        tr_inp = [(r["text_a"], r["text_b"]) for _, r in train.iterrows()]
        print(f"scoring train ({len(train):,}) for threshold tuning...",
              file=sys.stderr)
        tr_scores = model.predict(tr_inp, batch_size=128,
                                    show_progress_bar=True, convert_to_numpy=True)
        tr_scores = 1.0 / (1.0 + np.exp(-tr_scores))
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
    print(f"=== external transfer result ===")
    print(f"test: {os.path.basename(args.test_file)}  n={len(test)}  "
          f"positives={(test.label==1).sum()}")
    print(f"AUROC = {auroc:.3f}  AUPRC = {auprc:.3f}")
    print(f"F1@tau={thr:.3f}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
