"""Marquee-slice evaluation with LIVE scoring (replaces the coverage
caveats of eval_wikidata_slice.py).

The 302 match_quality=='both' Wikidata P156 pairs resolved to GLEIF,
minus those already in SuccessorEntity, are the high-profile M&A the
introduction motivates with (Sprint -> T-Mobile). They are absent from
the released pools, so scoring them tests generalization to (a)
survivorship-excluded events and (b) NON-adversarial candidate pools:
each anchor gets its true successor plus 10 random same-country
entities, i.e. no name-similar mining at all.

Every method is scored live on the exact same pools (no prediction-file
coverage gaps): token Jaccard and its negation, the full-record GBM
(retrained inline, seed 42), and both cross-encoders from their saved
checkpoints. Pools are drawn with several seeds; we report mean and
range across seeds.

Usage: python3 src/eval/eval_marquee_live.py [data] [--seeds 42 7 1337]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "method"))
from ranking import per_anchor_expected  # noqa: E402
from features_gbm import (featurize, load_entity_map_full,  # noqa: E402
                          normalize)


def build_pool(wd_new, by_country, rng):
    rows = []
    for _, r in wd_new.iterrows():
        pool = [l for l in by_country.get(r.gleif_succ_country, [])
                if l not in (r.gleif_pred_lei, r.gleif_succ_lei)]
        if len(pool) < 10:
            continue
        negs = list(rng.choice(pool, 10, replace=False))
        for c in [r.gleif_succ_lei] + negs:
            rows.append({"anchor_lei": r.gleif_pred_lei,
                         "candidate_lei": c,
                         "label": 1 if c == r.gleif_succ_lei else 0})
    return pd.DataFrame(rows)


def eh1(df):
    return float(np.mean([h[1] for _a, h, _e
                          in per_anchor_expected(df, ks=(1,))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1337])
    args = ap.parse_args()
    data = args.data

    wd = pq.read_table(os.path.join(data, "wikidata_marquee.parquet")
                       ).to_pandas()
    wd = wd[wd.match_quality == "both"].copy()
    sl = pq.read_table(os.path.join(data, "successor_links.parquet")
                       ).to_pandas()
    slp = set(zip(sl.pred_lei, sl.succ_lei))
    wd_new = wd[~wd.apply(
        lambda r: (r.gleif_pred_lei, r.gleif_succ_lei) in slp, axis=1)]
    print(f"marquee: {len(wd)} both-resolved, "
          f"{len(wd_new)} not in SuccessorEntity", file=sys.stderr)

    ent_c = pq.read_table(os.path.join(data, "entities.parquet"),
                          columns=["lei", "legal_addr_country"]
                          ).to_pandas().drop_duplicates("lei")
    by_country = {c: sub.lei.tolist()
                  for c, sub in ent_c.groupby("legal_addr_country")}

    # candidate LEIs across all seeds (superset for the entity map)
    pools = {sd: build_pool(wd_new, by_country,
                            np.random.default_rng(sd))
             for sd in args.seeds}
    leis = set()
    for df in pools.values():
        leis |= set(df.anchor_lei) | set(df.candidate_lei)

    # full-record GBM, retrained exactly as the paper's variant
    cols = ["anchor_lei", "candidate_lei", "label",
            "split_time", "split_event", "split_jurisdiction"]
    v2 = pq.read_table(os.path.join(data, "pairs_v2.parquet"),
                       columns=cols).to_pandas()
    tr = v2[(v2.split_time != "test") & (v2.split_event != "test")
            & (v2.split_jurisdiction != "test")]
    ent = load_entity_map_full(
        data, leis | set(tr.anchor_lei) | set(tr.candidate_lei))
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42
                                         ).fit(featurize(tr, ent, mode="full"),
                                               tr.label.values)
    print("GBM-full retrained", file=sys.stderr)

    import torch
    from sentence_transformers import CrossEncoder
    from train_cross_encoder import format_entity_fullrecord
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def fmt_basic(l):
        r = ent.get(l, {})
        n = (r.get("legal_name") or "").strip() or "(unknown)"
        c = (r.get("legal_addr_country") or "??").strip() or "??"
        f = (r.get("legal_form_code") or "??").strip() or "??"
        return f"{n} | jurisdiction: {c} | form: {f}"

    def fmt_full(l):
        r = ent.get(l, {})
        if not r:
            return "(unknown)"
        y = None
        try:
            yv = pd.to_datetime(r.get("entity_creation_date"), utc=True,
                                errors="coerce")
            if pd.notna(yv):
                y = str(yv.year)
        except Exception:
            pass
        return format_entity_fullrecord(r, year_str=y)

    ces = {}
    for key, path in [("CE name+state", "method_v2_crossenc_saved"),
                      ("CE full-record", "method_ce_fullrecord")]:
        mp = os.path.join(data, "models", path)
        ces[key] = CrossEncoder(mp, device=device) if os.path.isdir(mp) else None
        if ces[key] is None:
            print(f"  (no checkpoint for {key})", file=sys.stderr)

    results = {}
    for sd, df in pools.items():
        # token Jaccard both directions
        tok = lambda l: set(normalize(ent.get(l, {}).get("legal_name")).split())
        j = []
        for a, c in zip(df.anchor_lei, df.candidate_lei):
            ta, tb = tok(a), tok(c)
            u = ta | tb
            j.append(len(ta & tb) / len(u) if u else 0.0)
        t = df.copy(); t["score"] = j
        results.setdefault("token_set", []).append(eh1(t))
        t2 = df.copy(); t2["score"] = [-x for x in j]
        results.setdefault("anti-token_set", []).append(eh1(t2))

        g = df.copy()
        g["score"] = clf.predict_proba(featurize(g, ent, mode="full"))[:, 1]
        results.setdefault("GBM-full", []).append(eh1(g))

        for key, fmt in [("CE name+state", fmt_basic),
                         ("CE full-record", fmt_full)]:
            m = ces[key]
            if m is None:
                continue
            sc = m.predict([(fmt(a), fmt(c)) for a, c
                            in zip(df.anchor_lei, df.candidate_lei)],
                           batch_size=256, convert_to_numpy=True,
                           show_progress_bar=False)
            d = df.copy(); d["score"] = 1 / (1 + np.exp(-sc))
            results.setdefault(key, []).append(eh1(d))
        print(f"seed {sd} done ({df.anchor_lei.nunique()} anchors)",
              file=sys.stderr)

    n_anchors = pools[args.seeds[0]].anchor_lei.nunique()
    print(f"\nE[Hits@1] on the marquee slice "
          f"({n_anchors} anchors, gold + 10 random same-country):")
    for k, vals in results.items():
        print(f"  {k:16s} mean={np.mean(vals):.3f}  "
              f"range=[{min(vals):.3f}, {max(vals):.3f}]  (chance=0.091)")


if __name__ == "__main__":
    main()
