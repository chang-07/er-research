"""Generate figures for the TemporalLink paper.

Outputs PDFs (vector, for LaTeX) and PNG previews under paper/figures/.

Figures:
  fig1_event_year_histogram.pdf   stacked area of completed lifecycle events
                                   per year, 2014-2026, by top event types
  fig2_main_results.pdf           grouped bar chart: Hits@1 across baselines
                                   and method variants for the 3 split regimes
  fig3_per_event_type.pdf         Hits@1 broken down by anchor event type for
                                   the four most important baselines + method
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")

PALETTE = {
    "exact_name":         "#999999",
    "token_set":          "#bbbbbb",
    "tfidf_char":         "#888888",
    "sentence_encoder":   "#377eb8",
    "method_no_state":    "#a6cee3",
    "method_no_hard":     "#4daf4a",
    "method_full":        "#e41a1c",
    "method_v2_full":     "#984ea3",
    "method_v2_crossenc": "#ff7f00",
}
PRETTY = {
    "exact_name":         "exact-name",
    "token_set":          "token-set",
    "tfidf_char":         "TF-IDF char",
    "sentence_encoder":   "E5-small (zero-shot)",
    "method_no_state":    "method (no state)",
    "method_no_hard":     "method (no hard negs)",
    "method_full":        "method (full)",
    "method_v2_full":     "method v2 (bi)",
    "method_v2_crossenc": "method v2 (cross)",
}
SPLIT_PRETTY = {
    "split_time": "Held-out time",
    "split_event": "Held-out event type",
    "split_jurisdiction": "Held-out jurisdiction",
}

def fig1_histogram(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "event_year_histogram.csv"))
    df = df[(df["year"] >= 2014) & (df["year"] <= 2026)]
    # focus on the high-signal lifecycle types
    keep = ["MERGERS_AND_ACQUISITIONS", "CHANGE_LEGAL_NAME", "CHANGE_LEGAL_FORM",
            "ABSORPTION", "DEMERGER", "SPINOFF", "BREAKUP"]
    df = df[df["event_type"].isin(keep)]
    pv = (df.pivot(index="year", columns="event_type", values="count")
            .reindex(columns=keep, fill_value=0)
            .fillna(0).astype(int))
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bottom = np.zeros(len(pv.index))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
              "#ff7f00", "#ffff33", "#a65628"]
    for i, col in enumerate(keep):
        vals = pv[col].values
        ax.bar(pv.index, vals, bottom=bottom, label=col, color=colors[i],
               width=0.85, edgecolor="white", linewidth=0.4)
        bottom = bottom + vals
    ax.set_xlabel("Year (event effective date)")
    ax.set_ylabel("COMPLETED lifecycle events")
    ax.set_title("Lifecycle event volume in GLEIF (Level-1, 2026-05-19 snapshot)")
    ax.set_xticks(pv.index)
    ax.set_xticklabels(pv.index, rotation=0, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_event_year_histogram.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  wrote {p}", file=sys.stderr)


def fig2_main_results(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "eval_metrics.csv"))
    h1 = df[df["metric"] == "hits@1"].copy()
    pv = h1.pivot(index="baseline", columns="split", values="value")

    order = ["exact_name", "token_set", "tfidf_char", "sentence_encoder",
             "method_no_state", "method_no_hard", "method_full",
             "method_v2_full", "method_v2_crossenc"]
    pv = pv.reindex([b for b in order if b in pv.index])
    splits = ["split_time", "split_event", "split_jurisdiction"]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(len(pv.index))
    width = 0.27
    for i, s in enumerate(splits):
        ax.bar(x + (i - 1) * width, pv[s].values, width,
               label=SPLIT_PRETTY[s],
               color=["#377eb8", "#4daf4a", "#e41a1c"][i],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[b] for b in pv.index], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Hits@1 (test, per-anchor retrieval)")
    ax.set_title("Lifecycle-aware ER: method vs baselines, by held-out split")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(0.85, np.nanmax(pv.values) * 1.1))
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_main_results.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  wrote {p}", file=sys.stderr)


def fig3_per_event_type(data_dir, out_dir):
    pet = pd.read_csv(os.path.join(data_dir, "per_event_type_metrics.csv"))
    if pet.empty:
        return
    # Pick the most populous event types
    n_per = pet.groupby("event_type")["n_anchors"].first().sort_values(ascending=False)
    top_types = list(n_per.head(5).index)
    keep_baselines = ["sentence_encoder", "method_full",
                       "method_v2_full", "method_v2_crossenc"]
    sub = pet[pet["event_type"].isin(top_types) & pet["baseline"].isin(keep_baselines)]
    pv = sub.pivot(index="event_type", columns="baseline", values="hits1")
    keep_present = [b for b in keep_baselines if b in pv.columns]
    pv = pv.reindex(top_types)[keep_present]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(len(pv.index))
    width = 0.20
    colors = ["#377eb8", "#e41a1c", "#984ea3", "#ff7f00"][:len(keep_present)]
    for i, b in enumerate(pv.columns):
        ax.bar(x + (i - 1.5) * width, pv[b].values, width,
               label=PRETTY[b], color=colors[i],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    labels = [f"{t}\n(n={int(n_per[t])})" for t in pv.index]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Hits@1 (split_time test)")
    ax.set_title("Per-event-type Hits@1: where the method helps most")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig3_per_event_type.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  wrote {p}", file=sys.stderr)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing figures to {out_dir}", file=sys.stderr)
    fig1_histogram(args.data, out_dir)
    fig2_main_results(args.data, out_dir)
    fig3_per_event_type(args.data, out_dir)
    print("done.", file=sys.stderr)

if __name__ == "__main__":
    main()
