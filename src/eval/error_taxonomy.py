"""Build a named error taxonomy for cross-encoder failures, per split.

Sample 50 failure cases (positive ranked > 1 for its anchor) from each split's
test set. Auto-tag with heuristic categories, write data/error_taxonomy.csv.

Categories (mutually exclusive, top match wins):
  exact_normalized_name_collision  top-1 wrong shares anchor's normalized name
  fooled_by_parent_subsidiary       top-1 wrong is parent/subsidiary class
  acquirer_name_takeover            top-1 wrong contains the acquirer's name
                                    pattern (only on M&A/ABSORPTION anchors)
  substring_name_overlap            anchor name is substring of wrong's name
                                    (or vice-versa)
  no_shared_tokens_with_true         anchor and true successor share no tokens
  same_country_token_overlap         top-1 wrong shares >=2 tokens with anchor
  other                              residual
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
}

def normalize_name(name):
    if not name:
        return ""
    s = name.upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)


def categorize(r):
    n_a = (r["anchor_name"] or "").upper()
    n_t = (r["true_name"] or "").upper()
    n_w = (r["wrong_name"] or "").upper()
    nn_a, nn_t, nn_w = normalize_name(n_a), normalize_name(n_t), normalize_name(n_w)
    src = r["wrong_source"]

    # exact normalized collision (anchor == wrong after normalization)
    if nn_a and nn_a == nn_w:
        return "exact_normalized_name_collision"
    # parent/subsidiary
    if src == "parent_subsidiary":
        return "fooled_by_parent_subsidiary"
    # substring overlap
    if nn_a and nn_w and (nn_a in nn_w or nn_w in nn_a):
        return "substring_name_overlap"
    # no shared tokens between anchor and true successor — hardest case
    ta = set(nn_a.split()); tt = set(nn_t.split())
    if ta and tt and not (ta & tt):
        return "no_shared_tokens_with_true"
    # same-country token overlap with wrong
    tw = set(nn_w.split())
    if r["anchor_country"] == r["wrong_country"] and len(ta & tw) >= 2:
        return "same_country_token_overlap"
    # acquirer name takeover (M&A/absorption + name change away from anchor)
    if (r["anchor_event_type"] in ("MERGERS_AND_ACQUISITIONS", "ABSORPTION")
            and ta and tt and not (ta & tt)):
        return "acquirer_name_takeover"
    return "other"


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    preds = pq.read_table(os.path.join(data, "predictions",
                                        "method_v2_crossenc.parquet")).to_pandas()
    df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
    df["score"] = df["score"].fillna(0.0)

    ent = pq.read_table(os.path.join(data, "entities.parquet"),
                         columns=["lei", "legal_name", "legal_addr_country"]
                         ).to_pandas().drop_duplicates("lei")
    name_map = dict(zip(ent["lei"], ent["legal_name"].fillna("")))
    country_map = dict(zip(ent["lei"], ent["legal_addr_country"].fillna("")))

    SPLITS = [("split_time", "TIME"), ("split_event", "EVENT"),
              ("split_jurisdiction", "JURIS")]
    all_failures = []
    for split_col, split_name in SPLITS:
        test = df[df[split_col] == "test"].copy()
        failures = []
        for anchor, sub in test.groupby("anchor_lei"):
            if sub["label"].sum() == 0 or len(sub) < 2:
                continue
            ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
            pos_rows = ranked[ranked["label"] == 1]
            if pos_rows.empty:
                continue
            rank = int(ranked.index[ranked["label"] == 1][0]) + 1
            if rank == 1:
                continue
            pos_row = pos_rows.iloc[0]
            top_row = ranked.iloc[0]
            failures.append({
                "split": split_name,
                "anchor_lei": anchor,
                "anchor_name": name_map.get(anchor, ""),
                "anchor_country": country_map.get(anchor, ""),
                "anchor_event_type": pos_row["anchor_event_type"],
                "true_lei": pos_row["candidate_lei"],
                "true_name": name_map.get(pos_row["candidate_lei"], ""),
                "true_score": float(pos_row["score"]),
                "wrong_lei": top_row["candidate_lei"],
                "wrong_name": name_map.get(top_row["candidate_lei"], ""),
                "wrong_country": country_map.get(top_row["candidate_lei"], ""),
                "wrong_source": top_row["source"],
                "wrong_score": float(top_row["score"]),
                "rank": rank,
            })
        # sample 50 per split (seed=42)
        if len(failures) > 50:
            failures = pd.DataFrame(failures).sample(n=50, random_state=42).to_dict(orient="records")
        all_failures.extend(failures)

    out = pd.DataFrame(all_failures)
    out["error_category"] = out.apply(categorize, axis=1)

    # Per-split category distribution
    summary = out.groupby(["split", "error_category"]).size().unstack(fill_value=0)
    print("Cross-encoder failure category distribution (50 sampled per split):")
    print(summary.to_string())
    print()

    out_path = os.path.join(data, "cross_encoder_error_taxonomy.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote -> {out_path}")
    print(f"\nSample failures (first 3 per category):")
    for cat in sorted(out["error_category"].unique()):
        sub = out[out["error_category"] == cat].head(3)
        print(f"\n[{cat}] (n total = {(out['error_category']==cat).sum()})")
        for _, r in sub.iterrows():
            print(f"  anchor: '{r['anchor_name']}' ({r['anchor_country']}) [{r['anchor_event_type']}]")
            print(f"  true:   '{r['true_name']}' (rank={r['rank']}, score={r['true_score']:.2f})")
            print(f"  wrong:  '{r['wrong_name']}' (score={r['wrong_score']:.2f}, src={r['wrong_source']})")


if __name__ == "__main__":
    main()
