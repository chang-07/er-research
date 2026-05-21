"""Walk successor_links chains to produce transitive positives.

For every chain A -> B -> C in the successor graph, emit (A, C) as a
transitive positive. Same logic for longer chains (A -> B -> C -> D
yields (A,C), (A,D), (B,D), etc.).

This is one of the three positive types in the v1 plan that we didn't
materialize in Phase 1 because we only had direct successor links. With
the same source data (successor_links.parquet) we can derive transitive
positives at no cost.

The output also records the intermediate LEIs on each chain so that
downstream training code can do path-aware leakage filtering --- e.g.,
reject transitive (A, C) if any intermediate node B is a test-set
anchor or successor in any split regime.

Output: data/transitive_positives.parquet
  pred_lei, succ_lei, depth, classification = "transitive_chain",
  intermediate_leis (comma-separated string of LEIs strictly between
  pred and succ on the chain; empty string for depth=1, never written)
"""
import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-depth", type=int, default=4,
                    help="how deep to walk chains (depth 1 = direct, in successor_links already)")
    args = ap.parse_args()

    succ = pq.read_table(
        os.path.join(args.data, "successor_links.parquet"),
        columns=["pred_lei", "succ_lei", "classification"],
    ).to_pandas()
    direct = succ[succ["classification"] != "likely_administrative"].copy()
    print(f"loaded {len(direct):,} plausibly-real direct successor links",
          file=sys.stderr)

    # Build adjacency: pred -> set(succ)
    adj = defaultdict(set)
    for p, s in zip(direct["pred_lei"], direct["succ_lei"]):
        adj[p].add(s)

    # BFS each predecessor up to max_depth, tracking parent pointers so we
    # can reconstruct the chain for path-aware leakage filtering downstream.
    print(f"walking chains to depth {args.max_depth}...", file=sys.stderr)
    rows = []
    direct_pairs = set(zip(direct["pred_lei"], direct["succ_lei"]))
    for start in adj:
        frontier = [(start, 0)]
        parent = {start: None}    # node -> predecessor on the path from `start`
        while frontier:
            curr, d = frontier.pop()
            if d >= args.max_depth:
                continue
            for nxt in adj.get(curr, ()):
                if nxt in parent:
                    continue
                parent[nxt] = curr
                if d + 1 >= 2 and (start, nxt) not in direct_pairs:
                    # reconstruct intermediate nodes (strictly between start and nxt)
                    intermediates = []
                    node = curr
                    while node is not None and node != start:
                        intermediates.append(node)
                        node = parent[node]
                    intermediates.reverse()
                    rows.append({
                        "pred_lei": start,
                        "succ_lei": nxt,
                        "depth": d + 1,
                        "classification": "transitive_chain",
                        "intermediate_leis": ",".join(intermediates),
                    })
                frontier.append((nxt, d + 1))

    df = pd.DataFrame(rows).drop_duplicates(subset=["pred_lei", "succ_lei"], keep="first")
    print()
    print(f"transitive positives generated (not in direct successor_links): {len(df):,}")
    if not df.empty:
        print()
        print("Depth distribution:")
        print(df["depth"].value_counts().sort_index().to_string())

    out_path = os.path.join(args.data, "transitive_positives.parquet")
    df.to_parquet(out_path, compression="snappy", index=False)
    print(f"\nwrote -> {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
