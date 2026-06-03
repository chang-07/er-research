"""Evaluate Hits@1 / MRR on the fielded-retriever pair file.

For each test anchor, the pair file lists the top-K bi-encoder-retrieved
candidates + any true successors that weren't retrieved. We measure:
  - Recall@K (does the true successor appear in the bi-encoder top-K?)
  - Hits@1 (does the reranker's top-1 match the true successor?)
  - MRR (mean reciprocal rank of the top true successor in the reranker's list)

We restrict Hits@1 / MRR to anchors where the bi-encoder DID retrieve at
least one true successor in top-K. The recall@K number is what tells you
how often the bi-encoder fails the deployment pipeline before the
cross-encoder gets a chance.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def hits1_mrr_ci(df, n_boots=500, seed=42):
    rng = np.random.default_rng(seed)
    hits, ranks = [], []
    for anchor, sub in df.groupby("anchor_lei", sort=False):
        if sub["label"].sum() == 0 or len(sub) < 2:
            continue
        sub = sub.sample(frac=1.0, random_state=0)
        ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
        pi = ranked.index[ranked["label"] == 1]
        if len(pi) == 0:
            continue
        first = int(pi[0]) + 1
        hits.append(1 if first == 1 else 0)
        ranks.append(1.0 / first)
    if not hits:
        return (float("nan"),) * 6 + (0,)
    h = np.asarray(hits, dtype=float)
    m = np.asarray(ranks, dtype=float)
    bh = np.empty(n_boots); bm = np.empty(n_boots)
    for i in range(n_boots):
        idx = rng.integers(0, len(h), size=len(h))
        bh[i] = h[idx].mean()
        bm[i] = m[idx].mean()
    return (float(h.mean()),
            float(np.percentile(bh, 2.5)),
            float(np.percentile(bh, 97.5)),
            float(m.mean()),
            float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)),
            int(len(h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pairs", default="pairs_fielded.parquet")
    ap.add_argument("--preds", nargs="+", required=True,
                    help="prediction file stems to evaluate")
    args = ap.parse_args()

    pairs = pq.read_table(
        os.path.join(args.data, args.pairs)
    ).to_pandas()
    print(f"loaded {len(pairs):,} fielded pair rows", file=sys.stderr)

    splits = ("split_time", "split_event", "split_jurisdiction")
    short = {"split_time": "TIME", "split_event": "EVENT",
              "split_jurisdiction": "JURIS"}

    # ---- bi-encoder recall@K --------------------------------------------
    print("\n=== Bi-encoder Recall@20 (fielded retriever) ===")
    for split in splits:
        sub = pairs[pairs[split] == "test"]
        per_anc = sub.groupby("anchor_lei").apply(
            lambda g: ((g["label"] == 1) & (g["retrieval_rank"] > 0)).any()
        )
        n = len(per_anc)
        r = per_anc.mean() if n else float("nan")
        print(f"  {short[split]:<6s}: Recall@20 = {r*100:.1f}% "
              f"({int(per_anc.sum())}/{n} anchors)")

    # ---- score-driven reranking: cross-encoder + LLMs -------------------
    # Two views:
    #   (A) Conditional H@1: only anchors where bi-encoder retrieved
    #       truth in top-K; candidate set = top-K retrieved (truth was
    #       in there). Measures pure reranker quality.
    #   (B) End-to-end H@1: ALL test anchors; a hit requires
    #       truth-retrieved AND reranker picks it. The deployment metric.
    for stem in args.preds:
        ppath = os.path.join(args.data, "predictions", f"{stem}.parquet")
        if not os.path.exists(ppath):
            print(f"\nSKIP {stem}: file not found", file=sys.stderr)
            continue
        preds = pq.read_table(ppath).to_pandas()
        df = pairs.merge(preds[["anchor_lei", "candidate_lei", "score"]],
                          on=["anchor_lei", "candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0.0)

        print(f"\n=== Reranker: {stem} ===")
        for split in splits:
            test = df[df[split] == "test"]
            # Drop missed-truth fallback rows — those weren't actually retrieved
            test_retrieved = test[test["retrieval_rank"] > 0]

            # Conditional: per anchor, look at retrieved-only set
            cond_hits, cond_ranks = [], []
            for anc, sub in test_retrieved.groupby("anchor_lei", sort=False):
                if sub["label"].sum() == 0 or len(sub) < 2:
                    continue
                sub = sub.sample(frac=1.0, random_state=0)
                ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
                pi = ranked.index[ranked["label"] == 1]
                if len(pi) == 0:
                    continue
                first = int(pi[0]) + 1
                cond_hits.append(1 if first == 1 else 0)
                cond_ranks.append(1.0 / first)

            # End-to-end: per anchor in the full test set
            # A hit requires truth-retrieved AND cross-encoder ranks it #1
            e2e_hits, e2e_ranks = [], []
            for anc, sub_all in test.groupby("anchor_lei", sort=False):
                # Drop missed-truth from the candidate set for ranking
                sub_retrieved = sub_all[sub_all["retrieval_rank"] > 0]
                if len(sub_retrieved) < 1:
                    continue
                if sub_retrieved["label"].sum() == 0:
                    # truth was not retrieved -> guaranteed miss
                    e2e_hits.append(0)
                    e2e_ranks.append(0.0)
                    continue
                sub_retrieved = sub_retrieved.sample(frac=1.0, random_state=0)
                ranked = sub_retrieved.sort_values("score", ascending=False).reset_index(drop=True)
                pi = ranked.index[ranked["label"] == 1]
                first = int(pi[0]) + 1
                e2e_hits.append(1 if first == 1 else 0)
                e2e_ranks.append(1.0 / first)

            import numpy as np
            rng = np.random.default_rng(42)

            def boot(arr, n_boots=500):
                arr = np.asarray(arr, dtype=float)
                if len(arr) == 0:
                    return float("nan"), float("nan"), float("nan")
                bs = np.empty(n_boots)
                for i in range(n_boots):
                    idx = rng.integers(0, len(arr), size=len(arr))
                    bs[i] = arr[idx].mean()
                return (float(arr.mean()),
                        float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5)))

            ch, chlo, chhi = boot(cond_hits)
            cm, _, _ = boot(cond_ranks)
            eh, ehlo, ehhi = boot(e2e_hits)
            em, _, _ = boot(e2e_ranks)
            print(f"  {short[split]:<6s}  "
                  f"cond n={len(cond_hits):>4d} H@1 {ch:.3f} [{chlo:.3f},{chhi:.3f}] MRR {cm:.3f}  "
                  f"|  e2e n={len(e2e_hits):>4d} H@1 {eh:.3f} [{ehlo:.3f},{ehhi:.3f}] MRR {em:.3f}")


if __name__ == "__main__":
    main()
