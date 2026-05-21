"""B-cubed cluster-level precision/recall on TemporalLink positives.

Pair-level F1 is the standard ER metric but Binette (arXiv 2404.05622) and
the OpenSanctions Pairs paper both argue it's misleading for downstream
clustering / deduplication use cases. Cluster-level metrics (B-cubed) reward
methods that correctly group ALL records of a true entity together, not
just individual pairs.

Setup. The ground-truth clusters here are GLEIF successor-chain connected
components: e.g. A->B->C and A->D form a chain, and {A, B, C, D} is one
true cluster. We build them from the direct successor links plus the
mined transitive positives.

For each method, predicted clusters come from thresholded scores:
  union-find over (anchor, candidate) pairs with score >= tau,
  where tau is tuned on the train split to maximize pair F1.

B-cubed per record:
  P(r) = |true_cluster(r) ∩ pred_cluster(r)| / |pred_cluster(r)|
  R(r) = |true_cluster(r) ∩ pred_cluster(r)| / |true_cluster(r)|
B-cubed P/R over all records = mean over r.

Output: data/cluster_metrics.csv  (method, split, b3_precision, b3_recall, b3_f1)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            x, self.parent[x] = self.parent[x], root
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self):
        out = defaultdict(set)
        for x in self.parent:
            out[self.find(x)].add(x)
        return list(out.values())

    def cluster_of(self, x):
        if x not in self.parent:
            return {x}
        root = self.find(x)
        # collect everyone whose root is `root`
        return {y for y in self.parent if self.find(y) == root}


def true_clusters_from_positives(positives_df):
    """Union-find over (anchor, candidate) pairs with label=1 produces
    connected components = true entity clusters.
    """
    uf = UnionFind()
    for a, c in zip(positives_df["anchor_lei"], positives_df["candidate_lei"]):
        uf.union(a, c)
    return uf


def b3(pred_uf, true_uf, records):
    """B-cubed P/R over the given record universe. Records absent from both
    UFs are treated as singletons (cluster of themselves)."""
    # Pre-compute cluster membership lookups
    def cluster_of(uf, r):
        if r not in uf.parent:
            return frozenset({r})
        root = uf.find(r)
        return frozenset(y for y in uf.parent if uf.find(y) == root)

    # cache cluster lookups to avoid recomputing
    pred_clusters = {}
    true_clusters = {}
    for r in records:
        if r not in pred_clusters:
            pred_clusters[r] = cluster_of(pred_uf, r)
        if r not in true_clusters:
            true_clusters[r] = cluster_of(true_uf, r)

    p_sum, r_sum = 0.0, 0.0
    for r in records:
        pc, tc = pred_clusters[r], true_clusters[r]
        inter = len(pc & tc)
        p_sum += inter / len(pc)
        r_sum += inter / len(tc)
    n = len(records)
    p, r = p_sum / n, r_sum / n
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def tune_threshold(scores, labels):
    """Pick threshold maximizing pair F1 (cheap proxy for cluster F1)."""
    from sklearn.metrics import precision_recall_curve
    if len(np.unique(labels)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(labels, scores)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / np.where(p + r > 0, p + r, 1)
    if len(t) == 0:
        return 0.5
    return float(t[int(np.nanargmax(f1))])


def main():
    data = "data"
    pairs = pq.read_table(os.path.join(data, "pairs.parquet")).to_pandas()

    METHODS = ["sentence_encoder", "method_v2_full", "method_v2_crossenc"]
    SPLITS = ["split_time", "split_event", "split_jurisdiction"]

    rows = []
    for method in METHODS:
        path = os.path.join(data, "predictions", f"{method}.parquet")
        if not os.path.exists(path):
            continue
        preds = pq.read_table(path).to_pandas()
        df = pairs.merge(preds, on=["anchor_lei", "candidate_lei"], how="left")
        df["score"] = df["score"].fillna(0.0)
        # Clip cosine scores to [0,1]
        df["score"] = df["score"].clip(0, 1)

        for split in SPLITS:
            train = df[df[split] == "train"]
            test = df[df[split] == "test"]
            if len(train) == 0 or len(test) == 0:
                continue

            # threshold tuned on train
            tau = tune_threshold(train["score"].to_numpy(),
                                  train["label"].to_numpy())

            # true clusters from test positives
            test_pos = test[test["label"] == 1]
            true_uf = true_clusters_from_positives(test_pos)

            # predicted clusters: union-find over test pairs with score >= tau
            pred_uf = UnionFind()
            for a, c, s in zip(test["anchor_lei"], test["candidate_lei"],
                                 test["score"]):
                pred_uf.find(a); pred_uf.find(c)
                if s >= tau:
                    pred_uf.union(a, c)

            # record universe: every LEI appearing in test
            records = list(set(test["anchor_lei"]) | set(test["candidate_lei"]))
            p, r, f1 = b3(pred_uf, true_uf, records)
            rows.append({"method": method, "split": split,
                          "tau_tuned": round(tau, 4),
                          "b3_precision": round(p, 4),
                          "b3_recall": round(r, 4),
                          "b3_f1": round(f1, 4),
                          "n_records": len(records),
                          "n_true_clusters": len(true_uf.clusters())})
            print(f"{method:30s} {split:25s} tau={tau:.3f}  "
                  f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={len(records)}")

    out = pd.DataFrame(rows)
    path = os.path.join(data, "cluster_metrics.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
