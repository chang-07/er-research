"""Plot Hits@1 vs training-pair count for the bi-encoder (v2_full).

Reads predictions for method_v2_full_pct{10,25,50,75,100}.parquet,
computes Hits@1 per split, plots vs training pairs.

Output:
  data/data_efficiency.csv
  paper/figures/fig5_data_efficiency.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def hits1(d):
    hits, n = 0, 0
    for _, sub in d.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        hits += 1 if int(pi[0]) == 0 else 0
        n += 1
    return (hits / n) if n else 0.0


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    pcts = [10, 25, 50, 75, 100]
    total = 27_201  # transitive-augmented training positives

    rows = []
    for pct in pcts:
        path = os.path.join(data, "predictions",
                             f"method_v2_full_pct{pct}.parquet")
        if not os.path.exists(path):
            print(f"missing: {path} -- skip", file=sys.stderr)
            continue
        preds = pq.read_table(path).to_pandas()
        df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0.0)
        for split in ("split_time", "split_event", "split_jurisdiction"):
            test = df[df[split] == "test"]
            h = hits1(test)
            rows.append({"pct": pct,
                          "training_pairs": int(pct / 100 * total),
                          "split": split, "hits1": h})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(data, "data_efficiency.csv"), index=False)
    print(df_out.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    split_labels = {"split_time": "TIME",
                    "split_event": "EVENT",
                    "split_jurisdiction": "JURIS"}
    for split in ("split_time", "split_event", "split_jurisdiction"):
        sub = df_out[df_out["split"] == split].sort_values("training_pairs")
        ax.plot(sub["training_pairs"], sub["hits1"], marker="o",
                label=split_labels[split], lw=1.6, ms=6)
    ax.set_xlabel("Training pairs (positives + augmented transitives)")
    ax.set_ylabel("Hits@1 on test split")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Bi-encoder (E5-small v2\\_full) data efficiency")
    plt.tight_layout()
    out_pdf = "paper/figures/fig5_data_efficiency.pdf"
    plt.savefig(out_pdf, format="pdf", dpi=200)
    print(f"\nwrote -> {out_pdf}")


if __name__ == "__main__":
    main()
