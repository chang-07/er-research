"""Sample 300 random positives for human-in-the-loop label validation.

Outputs data/human_validation_pairs.jsonl with one pair per line:
  {id, anchor_name, anchor_country, anchor_form, succ_name, succ_country,
   succ_form, anchor_event_type, anchor_event_date, gleif_classification}

The annotator (author) reads each pair and judges whether the GLEIF
SuccessorEntity link is plausibly real. Output their labels to
data/human_validation_labels.jsonl as:
  {id, human_label}   # 1 (real), 0 (looks wrong), -1 (unsure)

Then `--report` computes agreement rate vs the GLEIF auto-classification
with a bootstrap 95% CI.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sample_pairs(data, n=300, seed=42):
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()
    succ = pq.read_table(os.path.join(data, "successor_links.parquet")).to_pandas()
    ent = pq.read_table(
        os.path.join(data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas().drop_duplicates("lei")
    ent_map = ent.set_index("lei").to_dict(orient="index")
    succ_class = dict(zip(zip(succ["pred_lei"], succ["succ_lei"]),
                            succ["classification"]))

    # Take POSITIVE pairs only
    pos = pairs[pairs["label"] == 1].copy()
    # Stratify by event type so all event types are represented
    by_evtype = []
    for ev, sub in pos.groupby("anchor_event_type"):
        # Sample proportionally with floor of 5 per event type
        k = max(5, int(round(n * len(sub) / len(pos))))
        k = min(k, len(sub))
        by_evtype.append(sub.sample(n=k, random_state=seed))
    sample = pd.concat(by_evtype, ignore_index=True)
    # Cap at exactly n if we over-sampled due to floors
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed).reset_index(drop=True)

    out = []
    for i, r in sample.iterrows():
        a = ent_map.get(r["anchor_lei"], {})
        b = ent_map.get(r["candidate_lei"], {})
        out.append({
            "id": int(i),
            "anchor_lei": r["anchor_lei"],
            "succ_lei": r["candidate_lei"],
            "anchor_name": a.get("legal_name", ""),
            "anchor_country": a.get("legal_addr_country", ""),
            "anchor_form": a.get("legal_form_code", ""),
            "succ_name": b.get("legal_name", ""),
            "succ_country": b.get("legal_addr_country", ""),
            "succ_form": b.get("legal_form_code", ""),
            "anchor_event_type": r["anchor_event_type"],
            "anchor_event_date": str(r["anchor_event_date"]) if pd.notna(r["anchor_event_date"]) else "",
            "gleif_classification": succ_class.get((r["anchor_lei"], r["candidate_lei"]), "unknown"),
        })
    out_path = os.path.join(data, "human_validation_pairs.jsonl")
    with open(out_path, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} pairs -> {out_path}")
    return out


def report(data):
    pairs_path = os.path.join(data, "human_validation_pairs.jsonl")
    labels_path = os.path.join(data, "human_validation_labels.jsonl")
    pairs = [json.loads(line) for line in open(pairs_path)]
    if not os.path.exists(labels_path):
        print(f"missing {labels_path} — annotate first")
        return
    labels = {row["id"]: row["human_label"]
              for row in (json.loads(l) for l in open(labels_path))}

    real, wrong, unsure = 0, 0, 0
    pair_by_id = {p["id"]: p for p in pairs}
    annotated_ids = [i for i in labels if i in pair_by_id]
    for i in annotated_ids:
        l = labels[i]
        if l == 1: real += 1
        elif l == 0: wrong += 1
        else: unsure += 1

    n = len(annotated_ids)
    if n == 0:
        print("no annotations found")
        return
    # Agreement with GLEIF's positive label (= "yes, this is a real successor")
    # GLEIF labels each pair label=1 in pairs.parquet; "wrong" = author disagrees
    valid = [i for i in annotated_ids if labels[i] != -1]
    if len(valid) > 0:
        agreement = sum(1 for i in valid if labels[i] == 1) / len(valid)
        # bootstrap CI
        rng = np.random.default_rng(42)
        boots = []
        agree_arr = np.array([labels[i] for i in valid])
        for _ in range(2000):
            sample = rng.choice(agree_arr, size=len(agree_arr), replace=True)
            boots.append(float((sample == 1).mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        agreement = float("nan"); lo = hi = float("nan")

    print(f"=== Author-validation pass on {n} sampled positives ===")
    print(f"  agree-with-GLEIF (label=real):    {real:3d} ({real/n*100:.1f}%)")
    print(f"  disagree-with-GLEIF (label=wrong): {wrong:3d} ({wrong/n*100:.1f}%)")
    print(f"  unsure:                           {unsure:3d} ({unsure/n*100:.1f}%)")
    print()
    print(f"  Agreement rate (excluding unsures, n={len(valid)}): "
          f"{agreement:.3f}  [95% bootstrap CI: {lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="instead of sampling, report agreement on existing labels")
    args = ap.parse_args()
    if args.report:
        report(args.data)
    else:
        sample_pairs(args.data, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
