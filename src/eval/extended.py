"""Extended evaluation for the Phase-4 paper write-up.

Outputs:
  data/per_event_type_metrics.csv  per-anchor-event-type retrieval breakdown
                                    for every baseline and method variant
  data/error_analysis.csv          100 sampled method_full failures on
                                    split_time test, with anchor/candidate
                                    record fields for qualitative tagging
  data/event_year_histogram.csv    counts of events by (year, event_type)
                                    for the dataset description figure
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PREDICTION_FILES = [
    "exact_name", "token_set", "tfidf_char", "sentence_encoder",
    "method_full", "method_no_state", "method_no_hard",
    "method_v2_full", "method_v2_crossenc",
    "method_v10_L6xL12",
]
SPLITS_TO_REPORT = ["split_time", "split_event", "split_jurisdiction"]

def per_event_type_hits1(test_df_with_score, top_n_types=12, n_boots=200, seed=42):
    """For each anchor, rank its candidates by score; bucket by event type.

    Returns DataFrame: (event_type, n_anchors, hits1, hits1_ci_lo, hits1_ci_hi,
                       hits3, mrr).

    Bootstrap CIs are computed by resampling anchors within an event type.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for evtype, sub in test_df_with_score.groupby("anchor_event_type"):
        hits1, hits3, rr = [], [], []
        for _, anc in sub.groupby("anchor_lei"):
            if anc["label"].sum() == 0 or len(anc) < 2:
                continue
            from ranking import expected_metrics
            hks, err = expected_metrics(anc["score"].values,
                                         anc["label"].values, ks=(1, 3))
            hits1.append(hks[1])
            hits3.append(hks[3])
            rr.append(err)
        if rr:
            arr = np.asarray(hits1, dtype=np.int8)
            n = len(arr)
            if n >= 5:
                boots = np.empty(n_boots)
                for i in range(n_boots):
                    idx = rng.integers(0, n, size=n)
                    boots[i] = arr[idx].mean()
                ci_lo = float(np.percentile(boots, 2.5))
                ci_hi = float(np.percentile(boots, 97.5))
            else:
                ci_lo = ci_hi = float("nan")
            rows.append({
                "event_type": evtype or "(unknown)",
                "n_anchors": n,
                "hits1": float(np.mean(hits1)),
                "hits1_ci_lo": ci_lo,
                "hits1_ci_hi": ci_hi,
                "hits3": float(np.mean(hits3)),
                "mrr": float(np.mean(rr)),
            })
    return pd.DataFrame(rows).sort_values("n_anchors", ascending=False).head(top_n_types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--error-baseline", default="method_full")
    ap.add_argument("--n-errors", type=int, default=100)
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label", "source",
                 "anchor_event_type", "anchor_event_date", "anchor_country",
                 "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()
    print(f"loaded {len(pairs):,} pairs", file=sys.stderr)

    # --- per-event-type breakdown for every (baseline, split) ------------
    rows = []
    for split_col in SPLITS_TO_REPORT:
        for bname in PREDICTION_FILES:
            ppath = os.path.join(args.data, "predictions", f"{bname}.parquet")
            if not os.path.exists(ppath):
                continue
            preds = pq.read_table(ppath).to_pandas()
            df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
            df["score"] = df["score"].fillna(0.0)
            test = df[df[split_col] == "test"]
            if len(test) == 0:
                continue
            breakdown = per_event_type_hits1(test)
            if breakdown.empty:
                continue
            breakdown["baseline"] = bname
            breakdown["split"] = split_col
            rows.append(breakdown)

    pet = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_pet = os.path.join(args.data, "per_event_type_metrics.csv")
    pet.to_csv(out_pet, index=False)
    print(f"wrote -> {out_pet}", file=sys.stderr)
    for split_col in SPLITS_TO_REPORT:
        sub = pet[pet["split"] == split_col]
        if sub.empty:
            continue
        print()
        print(f"Per-event-type Hits@1 ({split_col} test), top types:")
        pv = sub.pivot(index="event_type", columns="baseline", values="hits1").round(3)
        order = [b for b in PREDICTION_FILES if b in pv.columns]
        pv = pv[order]
        # also show n_anchors for context
        n_pv = sub.pivot(index="event_type", columns="baseline",
                          values="n_anchors").iloc[:, 0].astype(int)
        pv.insert(0, "n", n_pv)
        print(pv.to_string())

    # --- event-year histogram for the dataset figure ---------------------
    ev = pq.read_table(
        os.path.join(args.data, "events.parquet"),
        columns=["event_type", "event_status", "effective_date"],
    ).to_pandas()
    ev = ev[ev["event_status"] == "COMPLETED"]
    ev["year"] = pd.to_datetime(ev["effective_date"], utc=True, errors="coerce").dt.year
    hist = (ev.dropna(subset=["year"])
              .groupby(["year", "event_type"]).size()
              .reset_index(name="count"))
    out_hist = os.path.join(args.data, "event_year_histogram.csv")
    hist.to_csv(out_hist, index=False)
    print(f"\nwrote -> {out_hist}", file=sys.stderr)

    # --- error analysis: 100 method_full failures on split_time test -----
    err_path = os.path.join(args.data, "predictions", f"{args.error_baseline}.parquet")
    if not os.path.exists(err_path):
        print(f"\n(skipping error analysis: no {err_path})", file=sys.stderr)
        return
    preds = pq.read_table(err_path).to_pandas()
    df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
    df["score"] = df["score"].fillna(0.0)
    test = df[df["split_time"] == "test"].copy()

    # Per-anchor: figure out the score of the true positive vs top-1 candidate
    failures = []
    for anchor, sub in test.groupby("anchor_lei"):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.copy()
        sub_sorted = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pos_row = sub_sorted[sub_sorted["label"] == 1].iloc[0]
        top_row = sub_sorted.iloc[0]
        rank = int(sub_sorted.index[sub_sorted["label"] == 1][0]) + 1
        if rank > 1:
            # The model didn't rank the true successor first
            failures.append({
                "anchor_lei": anchor,
                "true_succ_lei": pos_row["candidate_lei"],
                "true_succ_score": float(pos_row["score"]),
                "top_wrong_lei": top_row["candidate_lei"],
                "top_wrong_source": top_row["source"],
                "top_wrong_score": float(top_row["score"]),
                "rank_of_true": rank,
                "n_candidates": len(sub_sorted),
                "anchor_event_type": pos_row["anchor_event_type"],
                "anchor_country": pos_row["anchor_country"],
            })
    fail_df = pd.DataFrame(failures)
    if not fail_df.empty:
        fail_df = fail_df.sample(n=min(args.n_errors, len(fail_df)), random_state=42)

    # Enrich with legal names for the qualitative review
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas().drop_duplicates(subset=["lei"])
    name_map = dict(zip(ent["lei"], ent["legal_name"].fillna("")))
    country_map = dict(zip(ent["lei"], ent["legal_addr_country"].fillna("")))
    form_map = dict(zip(ent["lei"], ent["legal_form_code"].fillna("")))

    if not fail_df.empty:
        fail_df["anchor_name"] = fail_df["anchor_lei"].map(name_map)
        fail_df["true_succ_name"] = fail_df["true_succ_lei"].map(name_map)
        fail_df["top_wrong_name"] = fail_df["top_wrong_lei"].map(name_map)
        fail_df["true_succ_country"] = fail_df["true_succ_lei"].map(country_map)
        fail_df["top_wrong_country"] = fail_df["top_wrong_lei"].map(country_map)

        # Heuristic auto-tagging — replaces manual labeling for v1
        def categorize(r):
            n_anchor = (r["anchor_name"] or "").lower()
            n_true = (r["true_succ_name"] or "").lower()
            n_wrong = (r["top_wrong_name"] or "").lower()
            same_anchor_wrong = n_anchor == n_wrong
            anchor_in_wrong = n_anchor and n_anchor in n_wrong
            wrong_in_anchor = n_wrong and n_wrong in n_anchor
            true_v_anchor_share = (
                bool(set(n_anchor.split()) & set(n_true.split())) if n_anchor and n_true else False)
            if same_anchor_wrong:
                return "exact_name_collision"
            if anchor_in_wrong or wrong_in_anchor:
                return "substring_name_overlap"
            if not true_v_anchor_share:
                return "anchor_and_true_share_no_tokens"  # really different name change
            if r["top_wrong_source"] == "parent_subsidiary":
                return "fooled_by_parent_sub"
            return "other"

        fail_df["error_category"] = fail_df.apply(categorize, axis=1)

    out_err = os.path.join(args.data, "error_analysis.csv")
    cols = ["error_category", "rank_of_true",
            "anchor_lei", "anchor_name", "anchor_country", "anchor_event_type",
            "true_succ_lei", "true_succ_name", "true_succ_country", "true_succ_score",
            "top_wrong_lei", "top_wrong_name", "top_wrong_country", "top_wrong_source", "top_wrong_score",
            "n_candidates"]
    if not fail_df.empty:
        fail_df = fail_df[cols]
    fail_df.to_csv(out_err, index=False)
    print(f"wrote -> {out_err}", file=sys.stderr)

    if not fail_df.empty:
        print()
        print(f"Error categories ({len(fail_df)} sampled failures of {args.error_baseline} on split_time test):")
        print(fail_df["error_category"].value_counts().to_string())
        print()
        print("Sample failures (first 5):")
        for _, r in fail_df.head(5).iterrows():
            print(f"  [{r['error_category']}] anchor='{r['anchor_name']}' ({r['anchor_country']})")
            print(f"     true successor: '{r['true_succ_name']}' ({r['true_succ_country']})  score={r['true_succ_score']:.3f}  rank={r['rank_of_true']}")
            print(f"     top-1 (wrong):  '{r['top_wrong_name']}' ({r['top_wrong_country']})  score={r['top_wrong_score']:.3f}  source={r['top_wrong_source']}")

if __name__ == "__main__":
    main()
