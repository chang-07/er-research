"""Helper: print the LaTeX row for method_v2_crossenc_L12 to paste into
Table 1 of the paper, plus the WA-L12 transfer numbers. Run after the
MiniLM-L-12 cross-encoder has been trained and scored.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


def hits1_mrr(d):
    hits, n = 0, 0
    rr = []
    for _, sub in d.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        from ranking import expected_metrics
        hks, err = expected_metrics(sub["score"].values,
                                     sub["label"].values, ks=(1,))
        hits += hks[1]
        n += 1
        rr.append(err)
    h = hits / n if n else 0.0
    m = float(np.mean(rr)) if rr else 0.0
    return h, m, n


def main():
    data = "data"
    L12_PATH = os.path.join(data, "predictions", "method_v2_crossenc_L12.parquet")
    if not os.path.exists(L12_PATH):
        print(f"WAIT: {L12_PATH} not present yet")
        sys.exit(1)

    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    preds = pq.read_table(L12_PATH).to_pandas()
    df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
    df["score"] = df["score"].fillna(0.0)

    print("=" * 70)
    print("method_v2_crossenc_L12 — paste into Table 1 of draft.tex:")
    print("=" * 70)
    out_cells = []
    for split in ("split_time", "split_event", "split_jurisdiction"):
        test = df[df[split] == "test"]
        if len(test) == 0:
            continue
        auroc = roc_auc_score(test.label, test.score)
        h, m, _ = hits1_mrr(test)
        out_cells.extend([f"{auroc:.3f}", f"{h:.3f}", f"{m:.3f}"])
    row = " & ".join(out_cells)
    print(f"    Cross-encoder (L-12, 33M)      & {row} \\\\")
    print()
    print("Latex-ready row above. Insert in Table 1 below the L-6 row.")

    # WA transfer (if scored)
    WA = os.path.join(data, "predictions",
                       "method_v2_crossenc_L12_on_walmart_amazon.parquet")
    if os.path.exists(WA):
        from sklearn.metrics import (precision_score, recall_score,
                                       f1_score, average_precision_score)
        wa = pq.read_table(WA).to_pandas()
        # Tune threshold on the L6 train numbers for fairness? We use 0.5 here.
        # If you want tuned, re-run score_external.py with --train-file.
        auroc = roc_auc_score(wa.label, wa.score)
        auprc = average_precision_score(wa.label, wa.score)
        pred = (wa.score >= 0.5).astype(int)
        p = precision_score(wa.label, pred, zero_division=0)
        r = recall_score(wa.label, pred, zero_division=0)
        f1 = f1_score(wa.label, pred, zero_division=0)
        print()
        print(f"WA transfer (L-12 zero-shot, threshold=0.5):")
        print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
