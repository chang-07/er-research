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

# ---- "full" mode: the structured GLEIF fields a production matcher ----
# would actually use on top of name similarity. Deliberately EXCLUDED
# as label leakage: has_successor / successor_lei (the labels are
# derived from them), plus registry bookkeeping (last_update_date,
# next_renewal_date, validation_sources).
FULL_ENTITY_COLS = [
    "lei", "legal_name", "other_names", "legal_addr_country",
    "legal_form_code", "legal_addr_city", "legal_addr_region",
    "legal_addr_postal_code", "hq_addr_country", "hq_addr_city",
    "entity_status", "registration_status", "category", "managing_lou",
    "entity_creation_date", "associated_entity_lei",
]

FULL_EXTRA_NAMES = [
    "othername_jac_ab", "othername_jac_ba",
    "same_city", "same_region", "same_postal", "postal_prefix3",
    "same_hq_country", "same_hq_city", "same_lou", "same_category",
    "cand_active", "anchor_inactive", "cand_reg_issued",
    "cand_reg_dead", "creation_delta_years", "cand_created_after",
    "assoc_link", "city_missing", "postal_missing",
]


def _year(v):
    try:
        y = int(str(v)[:4])
        return y if 1800 <= y <= 2030 else None
    except (ValueError, TypeError):
        return None


def _norm_field(v):
    return str(v).strip().upper() if v is not None and str(v).strip() else ""


def _best_othername_jac(tokens, other_names):
    best = 0.0
    if other_names is None:
        return best
    for name in other_names:
        tb = set(normalize(name).split())
        u = tokens | tb
        if u:
            best = max(best, len(tokens & tb) / len(u))
    return best


def pair_features_full(a, b):
    """Name features plus structured-field agreement for one pair."""
    base = pair_features(a, b)
    ta = set(normalize(a.get("legal_name")).split())
    tb = set(normalize(b.get("legal_name")).split())
    city_a, city_b = _norm_field(a.get("legal_addr_city")), _norm_field(b.get("legal_addr_city"))
    reg_a, reg_b = _norm_field(a.get("legal_addr_region")), _norm_field(b.get("legal_addr_region"))
    pc_a, pc_b = _norm_field(a.get("legal_addr_postal_code")), _norm_field(b.get("legal_addr_postal_code"))
    hqc_a, hqc_b = _norm_field(a.get("hq_addr_country")), _norm_field(b.get("hq_addr_country"))
    hqcity_a, hqcity_b = _norm_field(a.get("hq_addr_city")), _norm_field(b.get("hq_addr_city"))
    lou_a, lou_b = _norm_field(a.get("managing_lou")), _norm_field(b.get("managing_lou"))
    cat_a, cat_b = _norm_field(a.get("category")), _norm_field(b.get("category"))
    ya, yb = _year(a.get("entity_creation_date")), _year(b.get("entity_creation_date"))
    assoc = 1.0 if (_norm_field(a.get("associated_entity_lei")) == _norm_field(b.get("lei")) != ""
                    or _norm_field(b.get("associated_entity_lei")) == _norm_field(a.get("lei")) != "") else 0.0
    return base + [
        _best_othername_jac(ta, b.get("other_names")),
        _best_othername_jac(tb, a.get("other_names")),
        1.0 if (city_a and city_a == city_b) else 0.0,
        1.0 if (reg_a and reg_a == reg_b) else 0.0,
        1.0 if (pc_a and pc_a == pc_b) else 0.0,
        1.0 if (pc_a[:3] and pc_a[:3] == pc_b[:3]) else 0.0,
        1.0 if (hqc_a and hqc_a == hqc_b) else 0.0,
        1.0 if (hqcity_a and hqcity_a == hqcity_b) else 0.0,
        1.0 if (lou_a and lou_a == lou_b) else 0.0,
        1.0 if (cat_a and cat_a == cat_b) else 0.0,
        1.0 if _norm_field(b.get("entity_status")) == "ACTIVE" else 0.0,
        1.0 if _norm_field(a.get("entity_status")) == "INACTIVE" else 0.0,
        1.0 if _norm_field(b.get("registration_status")) == "ISSUED" else 0.0,
        1.0 if _norm_field(b.get("registration_status")) in ("RETIRED", "LAPSED", "DUPLICATE") else 0.0,
        float(abs(ya - yb)) if (ya and yb) else np.nan,
        (1.0 if yb > ya else 0.0) if (ya and yb) else np.nan,
        assoc,
        1.0 if not city_a or not city_b else 0.0,
        1.0 if not pc_a or not pc_b else 0.0,
    ]


def load_entity_map_full(data_dir, lei_set):
    import pyarrow.parquet as _pq
    ent = _pq.read_table(os.path.join(data_dir, "entities.parquet"),
                         columns=FULL_ENTITY_COLS).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].drop_duplicates(subset=["lei"], keep="first")
    return ent.set_index("lei", drop=False).to_dict(orient="index")


def featurize(df, ent_map, mode="name"):
    fn = pair_features_full if mode == "full" else pair_features
    width = len(FEATURE_NAMES) + (len(FULL_EXTRA_NAMES) if mode == "full" else 0)
    feats = np.empty((len(df), width), dtype=np.float32)
    for i, (al, cl) in enumerate(zip(df["anchor_lei"].values,
                                      df["candidate_lei"].values)):
        feats[i] = fn(ent_map.get(al, {}), ent_map.get(cl, {}))
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
    ap.add_argument("--seed", type=int, default=42,
                    help="random_state for HistGradientBoosting")
    ap.add_argument("--features", default="name", choices=["name", "full"],
                    help="'name' = the 12 released string+state features; "
                         "'full' adds address, status, LOU, category, "
                         "other-names and creation-date features "
                         "(successor fields excluded as label leakage)")
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
    ent_map = (load_entity_map_full(args.data, leis) if args.features == "full"
               else load_entity_map(args.data, leis))
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
    Xtr = featurize(train_df, ent_map, mode=args.features)
    ytr = train_df["label"].values

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=args.seed)
    clf.fit(Xtr, ytr)
    print("model fitted", file=sys.stderr)

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    for out_name, _ in pools.items():
        df = frames[out_name]
        X = featurize(df, ent_map, mode=args.features)
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
