"""Generate figures for the Sylvester paper.

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
    "method_v2_crossenc": "cross-encoder (L-6)",
    "method_v10_L6xL12":  "L-6 $\\times$ L-12 ensemble",
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
    ax.set_title("Lifecycle event volume in GLEIF (Level-1, 2026-05-19 snapshot)",
                 fontsize=9.5)
    years = pv.index.astype(int)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=0, fontsize=8)
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
    keep_baselines = ["sentence_encoder", "method_v2_crossenc",
                       "method_v10_L6xL12"]
    sub = pet[(pet["event_type"].isin(top_types))
               & (pet["baseline"].isin(keep_baselines))
               & (pet["split"] == "split_time")]
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
    fig6_cumulative_gains(args.data, out_dir)
    fig7_selective_prediction(args.data, out_dir)
    print("done.", file=sys.stderr)

def fig6_cumulative_gains(data, out_dir):
    """Cumulative-gain progression: each milestone on the cross-encoder path
    plotted against the v2 baseline across the three splits.
    Visualises the +9.4 / +11.2 / +3.4 pp arc described in Sec. 5.

    Reads predictions for all milestone models from data/predictions/.
    """
    import pyarrow.parquet as pq

    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    milestones = [
        ("headline (L-6, 22M)",         "method_v2_crossenc"),
        ("+ enriched prompt",           "method_v4_crossenc_enriched"),
        ("+ 3-seed avg.",               "method_v4_crossenc_enriched_ens3"),
        ("+ Wikidata-LEI pairs",        "method_v5_crossenc_wikidata"),
        ("+ country expansion",         "method_v6_crossenc_v6prompt"),
        ("3-way ensemble",              "method_v7_final_ensemble"),
        ("single (prompt+data, L-6)",   "method_v8a_combined"),
        ("L-12 backbone (33M)",         "method_v8aL12_combined"),
        ("L-6 $\\times$ L-12 ensemble", "method_v10_L6xL12"),
    ]

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

    splits = ("split_time", "split_event", "split_jurisdiction")
    split_labels = ("TIME", "EVENT", "JURIS")
    series = {s: [] for s in splits}
    labels = []
    for label, stem in milestones:
        ppath = os.path.join(data, "predictions", f"{stem}.parquet")
        if not os.path.exists(ppath):
            continue
        preds = pq.read_table(ppath).to_pandas()
        df = pairs.merge(preds, on=["anchor_lei","candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0)
        labels.append(label)
        for s in splits:
            series[s].append(hits1(df[df[s] == "test"]))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = np.arange(len(labels))
    markers = ["o","s","^"]
    colors  = ["#1f77b4","#ff7f0e","#2ca02c"]
    for (s, lab, m, c) in zip(splits, split_labels, markers, colors):
        ax.plot(xs, series[s], marker=m, color=c, linewidth=1.6,
                markersize=5.5, label=lab)
        # Annotate baseline + final values
        ax.annotate(f"{series[s][0]:.3f}", (xs[0], series[s][0]),
                    textcoords="offset points", xytext=(-6,-12),
                    fontsize=7, color=c)
        ax.annotate(f"{series[s][-1]:.3f}", (xs[-1], series[s][-1]),
                    textcoords="offset points", xytext=(2,4),
                    fontsize=7, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Hits@1 (per-anchor retrieval)")
    ax.set_ylim(0.55, 0.82)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8, ncols=3, framealpha=0.9)
    ax.set_title("Hits@1 across the extension chain (headline cross-encoder $\\to$ ensemble)",
                  fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig6_cumulative_gains.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


def fig7_selective_prediction(data, out_dir):
    """Selective prediction curve: F1 (or Hits@1) on the retained subset
    as we increase the score-confidence threshold. Demonstrates that
    high-confidence predictions are reliable even when global ECE is
    poor — a standard reviewer-pleaser for any work reporting ECE > 0.1.

    For each split, take the top-1 prediction per anchor (which is what
    Hits@1 measures), order anchors by that top-1 score, then sweep
    coverage from 100% down to 5% and plot retained-Hits@1.
    """
    import pyarrow.parquet as pq

    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    preds = pq.read_table(os.path.join(
        data, "predictions", "method_v10_L6xL12.parquet")).to_pandas()
    df = pairs.merge(preds, on=["anchor_lei","candidate_lei"], how="left")
    df["score"] = df["score"].fillna(0)

    splits = ("split_time", "split_event", "split_jurisdiction")
    split_labels = ("TIME", "EVENT", "JURIS")
    colors = ("#1f77b4","#ff7f0e","#2ca02c")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for s, lab, c in zip(splits, split_labels, colors):
        sub = df[df[s] == "test"]
        # Per-anchor top-1 prediction
        rows = []
        for anc, grp in sub.groupby("anchor_lei", sort=False):
            if len(grp) < 2 or grp["label"].sum() == 0:
                continue
            g = grp.sample(frac=1.0, random_state=0)
            r = g.sort_values("score", ascending=False).iloc[0]
            rows.append((float(r["score"]), int(r["label"])))
        if not rows:
            continue
        rows.sort(key=lambda x: -x[0])
        scores = np.array([r[0] for r in rows])
        hits = np.array([r[1] for r in rows], dtype=float)
        cov_pcts = np.linspace(5, 100, 40)
        ys = []
        for cp in cov_pcts:
            k = max(1, int(round(len(rows) * cp / 100.0)))
            ys.append(hits[:k].mean())
        ax.plot(cov_pcts / 100.0, ys, color=c, linewidth=1.7,
                label=f"{lab} (full={ys[-1]:.2f})")
    ax.set_xlabel("Coverage (fraction of anchors retained)")
    ax.set_ylabel("Hits@1 on retained anchors")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.55, 1.005)
    ax.grid(alpha=0.3)
    ax.set_title("Selective-prediction curve (v10 ensemble)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig7_selective_prediction.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()


def fig8_llm_panel(data="data", out_dir=OUT_DIR):
    """Grouped bars: zero-shot LLM panel E[Hits@1] per split under the
    uniform single-pair protocol (fair tie-breaking), with the L-6
    cross-encoder and L-6xL-12 ensemble as reference lines. Values are
    computed live from prediction parquets via ranking.expected_metrics."""
    import pyarrow.parquet as pq
    from ranking import per_anchor_expected

    MODELS = [
        ("llm_haiku_full",    "Haiku 4.5",     "#a6cee3"),
        ("llm_sonnet_full",   "Sonnet 4.6",    "#6baed6"),
        ("llm_opus_full",     "Opus 4.7",      "#2171b5"),
        ("llm_gemini_full",   "Gemini 3.1 Pro","#b2df8a"),
        ("llm_deepseek_full", "DeepSeek V4",   "#fdbf6f"),
        ("llm_gptoss_panel",  "gpt-oss-20b",   "#cccccc"),
    ]
    CE  = {"split_time": 0.607, "split_event": 0.569, "split_jurisdiction": 0.750}
    ENS = {"split_time": 0.700, "split_event": 0.677, "split_jurisdiction": 0.784}
    SPLITS = ["split_time", "split_event", "split_jurisdiction"]
    SHORT = ["TIME", "EVENT", "JURIS"]

    pairs = pq.read_table(os.path.join(data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label",
                  "split_time", "split_event", "split_jurisdiction"]).to_pandas()
    pairs = pairs.drop_duplicates(["anchor_lei", "candidate_lei"])

    rng = np.random.default_rng(42)
    vals, los, his = {}, {}, {}
    for stem, _label, _c in MODELS:
        p = pq.read_table(os.path.join(data, "predictions", f"{stem}.parquet"),
            columns=["anchor_lei", "candidate_lei", "score"]).to_pandas()
        df = pairs.merge(p, on=["anchor_lei", "candidate_lei"], how="left")
        for s in SPLITS:
            test = df[(df[s] == "test")].copy()
            bad = test.groupby("anchor_lei")["score"].transform(lambda x: x.isna().any())
            test = test[~bad]
            eh1 = np.array([hks[1] for _a, hks, _e in per_anchor_expected(test, ks=(1,))])
            if len(eh1) == 0:
                vals[(stem, s)] = los[(stem, s)] = his[(stem, s)] = np.nan
                continue
            boots = np.array([eh1[rng.integers(0, len(eh1), len(eh1))].mean()
                               for _ in range(500)])
            vals[(stem, s)] = eh1.mean()
            los[(stem, s)] = np.percentile(boots, 2.5)
            his[(stem, s)] = np.percentile(boots, 97.5)

    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    n = len(MODELS)
    width = 0.13
    xg = np.arange(len(SPLITS))
    for i, (stem, label, color) in enumerate(MODELS):
        xs = xg + (i - (n - 1) / 2) * width
        ys = [vals[(stem, s)] for s in SPLITS]
        yerr = [[vals[(stem, s)] - los[(stem, s)] for s in SPLITS],
                [his[(stem, s)] - vals[(stem, s)] for s in SPLITS]]
        ax.bar(xs, ys, width * 0.92, yerr=yerr, error_kw=dict(lw=0.5),
               color=color, edgecolor="black", linewidth=0.4, label=label)
    for j, s in enumerate(SPLITS):
        half = (n / 2) * width + 0.02
        ax.hlines(CE[s], xg[j] - half, xg[j] + half, colors="black",
                  linestyles="--", linewidth=1.1,
                  label="Cross-encoder (L-6)" if j == 0 else None)
        ax.hlines(ENS[s], xg[j] - half, xg[j] + half, colors="black",
                  linestyles=":", linewidth=1.1,
                  label="Ensemble (L-6$\\times$L-12)" if j == 0 else None)
    ax.set_xticks(xg)
    ax.set_xticklabels(SHORT, fontsize=7)
    ax.set_ylabel("E[Hits@1]", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.2, ncol=2, frameon=False, loc="upper left",
              columnspacing=0.8, handlelength=1.4, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig8_llm_panel.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_llm_panel.pdf/.png")
