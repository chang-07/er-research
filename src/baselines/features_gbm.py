"""Supervised feature-based matcher (the practitioner-default baseline).

Gradient-boosted trees over hand-engineered pair features: string
similarities (token Jaccard, char-3gram Jaccard, sequence ratio,
prefix overlap, length deltas) plus structured-field agreement
(jurisdiction, legal form). The Fellegi-Sunter lineage, modernized to
what production ER systems typically run.

Unlike the zero-shot similarity baselines, a SUPERVISED model can
learn this benchmark's anti-correlation (low name similarity predicts
a true successor on the curated pool) from the training labels — so
this row tests whether the cross-encoder's gap survives against a
learned exploit, not just an inverted heuristic.

Training follows the paper's no-leakage protocol: only pair rows in
`train` for ALL THREE split regimes. The fitted model scores all
three evaluation pools (benchmark, TF-IDF, fielded) so the off-pool
robustness story is measured for this baseline too.

Usage: python3 src/baselines/features_gbm.py --data data
Outputs: data/predictions/features_gbm{,_v3,_fielded}.parquet
"""
import argparse
import difflib
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
}


def normalize(name):
    if not name:
        return ""
    s = str(name).upper()
    for ch in [",", ".", "-", "&", "(", ")", "'", "/", "\\", '"', "*"]:
        s = s.replace(ch, " ")
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES and len(t) > 1)


def char3(s):
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s} if s else set()


def pair_features(a, b):
    """Feature vector for one (anchor_entity, candidate_entity) pair."""
    na, nb = normalize(a.get("legal_name")), normalize(b.get("legal_name"))
    ta, tb = set(na.split()), set(nb.split())
    u = ta | tb
    jac = len(ta & tb) / len(u) if u else 0.0
    c3a, c3b = char3(na.replace(" ", "")), char3(nb.replace(" ", ""))
    cu = c3a | c3b
    c3j = len(c3a & c3b) / len(cu) if cu else 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio() if (na and nb) else 0.0
    pref = 0
    for x, y in zip(na, nb):
        if x != y:
            break
        pref += 1
    mx = max(len(na), len(nb), 1)
    ca, cb = a.get("legal_addr_country") or "", b.get("legal_addr_country") or ""
    fa, fb = a.get("legal_form_code") or "", b.get("legal_form_code") or ""
    return [
        jac, c3j, seq, pref / mx,
        len(ta & tb), abs(len(ta) - len(tb)),
        abs(len(na) - len(nb)) / mx,
        1.0 if (na and na == nb) else 0.0,
        1.0 if (ca and ca == cb) else 0.0,
        1.0 if (fa and fa == fb) else 0.0,
        1.0 if not ca or not cb else 0.0,
        1.0 if not fa or not fb else 0.0,
    ]


FEATURE_NAMES = ["tok_jac", "char3_jac", "seq_ratio", "prefix_frac",
                 "shared_tok", "tok_delta", "len_delta_frac", "exact_norm",
                 "same_country", "same_form", "country_missing", "form_missing"]


def featurize(df, ent_map):
    feats = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (al, cl) in enumerate(zip(df["anchor_lei"].values,
                                      df["candidate_lei"].values)):
        feats[i] = pair_features(ent_map.get(al, {}), ent_map.get(cl, {}))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--train-pairs", default="pairs.parquet",
                    help="pool whose no-leakage rows the GBM trains on; "
                         "pairs_v3.parquet reproduces the trn-TF-IDF "
                         "variant reported in the paper (Sec 5.4)")
    ap.add_argument("--out-suffix", default="",
                    help="appended to output stems, e.g. _trnv3")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from llm_pairwise import load_entity_map

    sfx = args.out_suffix
    pools = {
        f"features_gbm{sfx}": "pairs.parquet",
        f"features_gbm{sfx}_v3": "pairs_v3.parquet",
        f"features_gbm{sfx}_fielded": "pairs_fielded.parquet",
    }
    cols = ["anchor_lei", "candidate_lei", "label",
            "split_time", "split_event", "split_jurisdiction"]
    frames = {}
    leis = set()
    for out_name, fname in pools.items():
        p = os.path.join(args.data, fname)
        frames[out_name] = pq.read_table(p, columns=cols).to_pandas()
        leis |= set(frames[out_name]["anchor_lei"])
        leis |= set(frames[out_name]["candidate_lei"])
        print(f"loaded {fname}: {len(frames[out_name]):,} rows", file=sys.stderr)

    bench = pq.read_table(os.path.join(args.data, args.train_pairs),
                          columns=cols).to_pandas()
    leis |= set(bench["anchor_lei"]) | set(bench["candidate_lei"])
    ent_map = load_entity_map(args.data, leis)
    print(f"entity map: {len(ent_map):,}", file=sys.stderr)

    # ---- train on the no-leakage rows of --train-pairs --------------------
    # No-leakage = not TEST in any regime ("excl" rows never reach a test
    # set) — same criterion as the neural models' training positives.
    train_mask = ((bench["split_time"] != "test")
                   & (bench["split_event"] != "test")
                   & (bench["split_jurisdiction"] != "test"))
    train_df = bench[train_mask]
    print(f"training rows (non-test in ALL three regimes): {len(train_df):,} "
          f"({int(train_df['label'].sum()):,} pos)", file=sys.stderr)
    Xtr = featurize(train_df, ent_map)
    ytr = train_df["label"].values

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    clf.fit(Xtr, ytr)
    print("model fitted", file=sys.stderr)

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    for out_name, _ in pools.items():
        df = frames[out_name]
        X = featurize(df, ent_map)
        score = clf.predict_proba(X)[:, 1]
        out = pd.DataFrame({"anchor_lei": df["anchor_lei"],
                            "candidate_lei": df["candidate_lei"],
                            "score": score.astype(float)})
        out = out.drop_duplicates(["anchor_lei", "candidate_lei"])
        out_path = os.path.join(out_dir, f"{out_name}.parquet")
        out.to_parquet(out_path, compression="snappy", index=False)
        print(f"wrote {out_path} ({len(out):,})", file=sys.stderr)


if __name__ == "__main__":
    main()
