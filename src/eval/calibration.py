"""Calibration analysis for the cross-encoder.

Computes ECE, Brier, and a 10-bin reliability diagram. Writes:
  data/calibration_metrics.csv
  paper/figures/fig4_calibration.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def reliability_table(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(scores, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lo": bins[b],
            "bin_hi": bins[b + 1],
            "n": n,
            "mean_score": float(scores[mask].mean()),
            "observed_pos_rate": float(labels[mask].mean()),
        })
    df = pd.DataFrame(rows)
    n_total = len(scores)
    ece = float(((df["n"] / n_total) *
                  (df["observed_pos_rate"] - df["mean_score"]).abs()).sum())
    brier = float(((labels - scores) ** 2).mean())
    return df, ece, brier


def plot(df_per_method, out_pdf):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    markers = ["o", "s", "^"]
    for (label, df), m in zip(df_per_method.items(), markers):
        ax.plot(df["mean_score"], df["observed_pos_rate"], marker=m,
                label=label, lw=1.5, ms=6)
    ax.set_xlabel("Mean predicted score (per bin)")
    ax.set_ylabel("Observed positive rate (per bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Reliability diagram (test split, 10 bins)")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", dpi=200)
    print(f"wrote -> {out_pdf}", file=sys.stderr)


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    test_mask = ((pairs["split_time"] == "test")
                 | (pairs["split_event"] == "test")
                 | (pairs["split_jurisdiction"] == "test"))
    test_keys = pairs[test_mask][["anchor_lei", "candidate_lei", "label"]] \
                    .drop_duplicates(subset=["anchor_lei", "candidate_lei"])

    METHODS = [
        ("cross-encoder v2", "method_v2_crossenc"),
        ("bi-encoder v2_full", "method_v2_full"),
        ("E5-small zero-shot", "sentence_encoder"),
    ]

    metric_rows = []
    per_method_curves = {}
    for label, stem in METHODS:
        path = os.path.join(data, "predictions", f"{stem}.parquet")
        if not os.path.exists(path):
            continue
        preds = pq.read_table(path).to_pandas()
        df = test_keys.merge(preds, on=["anchor_lei", "candidate_lei"],
                              how="left")
        df["score"] = df["score"].fillna(0.0)
        # Clip bi-encoder cosine scores to [0,1] (raw cosine can be negative)
        if stem != "method_v2_crossenc":
            df["score"] = df["score"].clip(0, 1)
        rel, ece, brier = reliability_table(
            df["score"].to_numpy(),
            df["label"].to_numpy().astype(float),
        )
        per_method_curves[label] = rel
        metric_rows.append({
            "method": label, "ece_10bin": round(ece, 4),
            "brier": round(brier, 4),
        })
        print(f"{label}: ECE={ece:.3f} Brier={brier:.3f}")

    pd.DataFrame(metric_rows).to_csv(
        os.path.join(data, "calibration_metrics.csv"), index=False)

    os.makedirs("paper/figures", exist_ok=True)
    plot(per_method_curves, "paper/figures/fig4_calibration.pdf")


if __name__ == "__main__":
    main()
