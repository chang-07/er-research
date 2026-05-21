"""Cross-encoder method: fine-tune a small joint encoder on (anchor, candidate)
pairs for the lifecycle-aware ER task.

A cross-encoder takes both records as joint input and produces a single
relevance score, unlike the bi-encoder which embeds each side independently
and uses cosine. Cross-encoders are typically 2-5 pp more accurate but
quadratically more expensive at inference --- well suited as a second-stage
re-ranker over a bi-encoder's top-K.

We train on the same positives as the bi-encoder + a sample of v2 hard
negs, and score ALL pairs in pairs.parquet so the result drops into the
standard eval pipeline.

Defaults:
  base: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~80 sentences/s on CPU)
  output: data/predictions/method_v2_crossenc.parquet
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

LEGAL_SUFFIXES = {
    "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "PLC", "PC",
    "SA", "SAS", "GMBH", "AG", "BV", "NV", "AB", "OY", "CO", "COMPANY",
    "GROUP", "HOLDINGS", "HOLDING", "PARTNERSHIP", "LP", "LLP",
    "TRUST", "FUND", "LTDA", "PTY", "BHD", "TBK", "SRL", "SPA", "SPRL",
    "THE", "AND", "OF",
}

def format_entity(name, country, form):
    name = (name or "").strip() or "(unknown)"
    country = (country or "??").strip() or "??"
    form = (form or "??").strip() or "??"
    return f"{name} | jurisdiction: {country} | form: {form}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--negs-per-pos", type=int, default=4,
                    help="how many hard negs to pair with each positive in training")
    ap.add_argument("--max-train-pairs", type=int, default=None)
    ap.add_argument("--out-name", default="method_v2_crossenc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-negs", default="hard_negatives_v2.parquet",
                    help="hard-neg parquet filename under --data for TRAINING")
    ap.add_argument("--train-pairs", default="pairs.parquet",
                    help="pairs parquet for TRAINING-set positives & split assignments")
    ap.add_argument("--extra-pairs", default="",
                    help="comma-separated list of EXTRA pairs files to score on. "
                         "Each produces <out-name>_on_<stem>.parquet predictions.")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)
    rng = np.random.default_rng(args.seed)

    # ---- load data --------------------------------------------------------
    print(f"loading {args.train_pairs}...", file=sys.stderr)
    pairs = pq.read_table(
        os.path.join(args.data, args.train_pairs),
        columns=["anchor_lei", "candidate_lei", "label",
                 "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()
    extra_pair_paths = [p.strip() for p in args.extra_pairs.split(",") if p.strip()]
    extra_pairs_dfs = {}
    for p in extra_pair_paths:
        pp = pq.read_table(
            os.path.join(args.data, p),
            columns=["anchor_lei", "candidate_lei"],
        ).to_pandas()
        extra_pairs_dfs[p] = pp
        print(f"  extra-eval {p}: {len(pp):,} rows", file=sys.stderr)
    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])
    for pp in extra_pairs_dfs.values():
        lei_set |= set(pp["anchor_lei"]) | set(pp["candidate_lei"])
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas().drop_duplicates(subset=["lei"], keep="first")
    ent = ent[ent["lei"].isin(lei_set)].reset_index(drop=True)
    ent_map = ent.set_index("lei").to_dict(orient="index")
    print(f"  {len(pairs):,} pair rows / {len(ent):,} entities", file=sys.stderr)

    def text_for(lei):
        r = ent_map.get(lei)
        if not r:
            return "(unknown)"
        return format_entity(r.get("legal_name"), r.get("legal_addr_country"),
                              r.get("legal_form_code"))

    # ---- training set (no leakage) ----------------------------------------
    pos = pairs[pairs["label"] == 1]
    is_safe = ((pos["split_time"] != "test")
               & (pos["split_event"] != "test")
               & (pos["split_jurisdiction"] != "test"))
    pos_train = pos[is_safe].copy()
    print(f"  train positives: {len(pos_train):,}", file=sys.stderr)

    # transitive positives augmentation, with path-aware leakage filter:
    # reject (A, C) if any node on the A->...->C chain is a test anchor or
    # test successor in any split regime.
    trans_path = os.path.join(args.data, "transitive_positives.parquet")
    if os.path.exists(trans_path):
        trans_cols = ["pred_lei", "succ_lei"]
        schema = pq.read_table(trans_path).schema.names
        if "intermediate_leis" in schema:
            trans_cols.append("intermediate_leis")
        trans = pq.read_table(trans_path, columns=trans_cols).to_pandas()
        test_anchors = set(pos[~is_safe]["anchor_lei"])
        test_succs = set(pos[~is_safe]["candidate_lei"])
        test_lei = test_anchors | test_succs
        # Endpoint filter
        mask = (~trans["pred_lei"].isin(test_anchors)
                & ~trans["succ_lei"].isin(test_anchors)
                & ~trans["succ_lei"].isin(test_succs))
        if "intermediate_leis" in trans.columns:
            def path_safe(s):
                if not s:
                    return True
                return not any(x in test_lei for x in s.split(","))
            inter_safe = trans["intermediate_leis"].fillna("").map(path_safe)
            pre_count = int(mask.sum())
            mask &= inter_safe
            post_count = int(mask.sum())
            print(f"  path-aware filter dropped {pre_count - post_count:,} "
                  f"transitive positives with test-leaking intermediates",
                  file=sys.stderr)
        trans = trans[mask]
        trans = trans.rename(columns={"pred_lei": "anchor_lei",
                                       "succ_lei": "candidate_lei"})
        trans["label"] = 1
        for col in pos_train.columns:
            if col not in trans.columns:
                trans[col] = None
        trans = trans[pos_train.columns]
        pos_train = pd.concat([pos_train, trans], ignore_index=True)
        print(f"  added transitives -> {len(pos_train):,}", file=sys.stderr)

    if args.max_train_pairs:
        pos_train = pos_train.sample(n=min(len(pos_train), args.max_train_pairs),
                                      random_state=args.seed)
        print(f"  capped to {len(pos_train):,}", file=sys.stderr)

    # hard negs for training
    hn_path = os.path.join(args.data, args.hard_negs)
    hard = pq.read_table(hn_path,
                          columns=["pred_lei", "neg_lei"]).to_pandas()
    hard = hard[hard["pred_lei"].isin(pos_train["anchor_lei"].unique())]
    hard_by_pred = hard.groupby("pred_lei")["neg_lei"].apply(list).to_dict()
    print(f"  hard negs for {len(hard_by_pred):,} predecessors", file=sys.stderr)

    # ---- training triples: positive + K negs ------------------------------
    print(f"building training examples ({args.negs_per_pos} negs/pos)...",
          file=sys.stderr)
    train_examples = []
    for _, row in pos_train.iterrows():
        pred, succ = row["anchor_lei"], row["candidate_lei"]
        anchor_text = text_for(pred)
        train_examples.append(InputExample(
            texts=[anchor_text, text_for(succ)], label=1.0))
        negs = hard_by_pred.get(pred, [])
        if negs:
            sample_idx = rng.choice(len(negs),
                                      size=min(args.negs_per_pos, len(negs)),
                                      replace=False)
            for j in sample_idx:
                train_examples.append(InputExample(
                    texts=[anchor_text, text_for(negs[j])], label=0.0))
    print(f"  {len(train_examples):,} training examples "
          f"({sum(1 for e in train_examples if e.label==1.0):,} pos / "
          f"{sum(1 for e in train_examples if e.label==0.0):,} neg)",
          file=sys.stderr)

    # ---- train -----------------------------------------------------------
    print(f"loading base model {args.base_model}...", file=sys.stderr)
    model = CrossEncoder(args.base_model, num_labels=1, device=device)
    train_dl = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    print(f"training for {args.epochs} epoch(s) at lr={args.lr}...",
          file=sys.stderr)
    t0 = time.time()
    model_out_dir = os.path.join(args.data, "models", args.out_name)
    model.fit(
        train_dataloader=train_dl,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=model_out_dir,
    )
    # Newer sentence-transformers versions don't always honor output_path on
    # CrossEncoder.fit(); explicit save guarantees the checkpoint lands.
    os.makedirs(model_out_dir, exist_ok=True)
    model.save(model_out_dir)
    print(f"  trained in {time.time()-t0:.1f}s, saved -> {model_out_dir}",
          file=sys.stderr)

    # ---- score the training-pairs file (headline preds) ------------------
    preds_dir = os.path.join(args.data, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    def score_and_write(df_pairs, out_name, label):
        print(f"scoring {len(df_pairs):,} pairs ({label})...", file=sys.stderr)
        t0 = time.time()
        inp = [(text_for(a), text_for(c))
               for a, c in zip(df_pairs["anchor_lei"], df_pairs["candidate_lei"])]
        scores = model.predict(inp, batch_size=128, show_progress_bar=True,
                                convert_to_numpy=True)
        # Sigmoid to bring logits to [0,1]
        scores = 1.0 / (1.0 + np.exp(-scores))
        print(f"  scored in {time.time()-t0:.1f}s "
              f"(mean={float(scores.mean()):.3f}, "
              f"median={float(np.median(scores)):.3f})", file=sys.stderr)
        out_path = os.path.join(preds_dir, f"{out_name}.parquet")
        out_df = pd.DataFrame({"anchor_lei": df_pairs["anchor_lei"],
                                "candidate_lei": df_pairs["candidate_lei"],
                                "score": scores.astype(float)})
        out_df.to_parquet(out_path, compression="snappy", index=False)
        print(f"wrote -> {out_path}", file=sys.stderr)

    score_and_write(pairs, args.out_name, args.train_pairs)

    # ---- additionally score each extra pairs file ------------------------
    for extra_name, extra_df in extra_pairs_dfs.items():
        stem = os.path.splitext(extra_name)[0]
        out_name = f"{args.out_name}_on_{stem}"
        score_and_write(extra_df, out_name, extra_name)

if __name__ == "__main__":
    main()
