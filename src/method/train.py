"""Train a lifecycle-aware bi-encoder via supervised contrastive learning.

Fine-tunes E5-small-v2 from successor-link positives in TemporalLink v1
plus the hard negatives mined in Phase 1B. The entity-state-at-time signal
(jurisdiction, legal form, optional listed-exchange and city) is folded into
the text prompt rather than a separate vector head — simpler and works well
for the v1 method paper.

Variants:
  full       state features + hard negatives + fine-tuning (default)
  no_state   only legal_name in the text (ablates state features)
  no_hard    no explicit hard negatives, just in-batch (ablates Phase 1B)
  zero_shot  skip training, just score (sanity — should match sentence_encoder)

For each variant, this script:
  1. Builds (anchor, positive, hard_negative) triples from pairs.parquet,
     restricted to positives that are in the train set of all three split
     regimes (no leakage onto any test split).
  2. Fine-tunes with MultipleNegativesRankingLoss using both the explicit
     hard negative and the in-batch positives of other anchors as negatives.
  3. Scores all 125K pairs in pairs.parquet and writes
     data/predictions/method_<variant>.parquet.

Default device is MPS (Apple Silicon) if available, else CPU.
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

VARIANTS = ("full", "no_state", "no_hard", "zero_shot", "v2_full")

def format_entity(name, country, form, variant):
    name = (name or "").strip() or "(unknown)"
    if variant == "no_state":
        return f"query: {name}"
    country = (country or "??").strip() or "??"
    form = (form or "??").strip() or "??"
    return f"query: {name} | jurisdiction: {country} | form: {form}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--variant", choices=VARIANTS, default="full")
    ap.add_argument("--base-model", default="intfloat/e5-small-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-train-pairs", type=int, default=None,
                    help="cap training positives (for quick debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)

    # ---- load data --------------------------------------------------------
    print("loading parquet...", file=sys.stderr)
    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label",
                 "split_time", "split_event", "split_jurisdiction"],
    ).to_pandas()
    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].drop_duplicates(subset=["lei"], keep="first").reset_index(drop=True)
    ent_map = ent.set_index("lei").to_dict(orient="index")
    print(f"  {len(pairs):,} pair rows / {len(ent):,} entities", file=sys.stderr)

    def text_for(lei):
        r = ent_map.get(lei)
        if not r:
            return "query: (unknown)"
        return format_entity(r.get("legal_name"), r.get("legal_addr_country"),
                              r.get("legal_form_code"), args.variant)

    # ---- training set: positives in train of all three splits ------------
    pos = pairs[pairs["label"] == 1]
    is_safe = ((pos["split_time"] != "test")
               & (pos["split_event"] != "test")
               & (pos["split_jurisdiction"] != "test"))
    pos_train = pos[is_safe].copy()
    print(f"  train positives (no-test in any split): {len(pos_train):,}",
          file=sys.stderr)

    # v2: augment with transitive successor-chain positives
    if args.variant == "v2_full":
        trans_path = os.path.join(args.data, "transitive_positives.parquet")
        if os.path.exists(trans_path):
            trans = pq.read_table(trans_path,
                                   columns=["pred_lei", "succ_lei"]).to_pandas()
            # only keep transitive positives whose anchor and successor are
            # both in our train-safe pool (defense against leakage)
            train_anchors = set(pos_train["anchor_lei"])
            test_anchors = set(pos[~is_safe]["anchor_lei"])
            test_succs = set(pos[~is_safe]["candidate_lei"])
            trans = trans[trans["pred_lei"].isin(train_anchors)
                          & ~trans["succ_lei"].isin(test_anchors)
                          & ~trans["succ_lei"].isin(test_succs)]
            trans = trans.rename(columns={"pred_lei": "anchor_lei",
                                          "succ_lei": "candidate_lei"})
            trans["label"] = 1
            trans["source"] = "transitive_chain"
            for col in pos_train.columns:
                if col not in trans.columns:
                    trans[col] = None
            trans = trans[pos_train.columns]
            pre = len(pos_train)
            pos_train = pd.concat([pos_train, trans], ignore_index=True)
            print(f"  added {len(pos_train)-pre:,} transitive positives "
                  f"(total now {len(pos_train):,})", file=sys.stderr)

    if args.max_train_pairs:
        pos_train = pos_train.sample(n=min(len(pos_train), args.max_train_pairs),
                                      random_state=args.seed)
        print(f"  capped to {len(pos_train):,}", file=sys.stderr)

    # ---- hard negatives ---------------------------------------------------
    if args.variant in ("full", "no_state", "v2_full"):
        # v2 uses hard_negatives_v2.parquet (E5-cosine mining); others use v1.
        hn_path = (os.path.join(args.data, "hard_negatives_v2.parquet")
                   if args.variant == "v2_full"
                   else os.path.join(args.data, "hard_negatives.parquet"))
        print(f"  loading hard negs from {os.path.basename(hn_path)}",
              file=sys.stderr)
        hard = pq.read_table(hn_path,
            columns=["pred_lei", "neg_lei", "negative_class"],
        ).to_pandas()
        # restrict to negatives for our training predecessors
        hard = hard[hard["pred_lei"].isin(pos_train["anchor_lei"].unique())]
        hard_by_pred = hard.groupby("pred_lei")["neg_lei"].apply(list).to_dict()
        print(f"  hard-neg LEIs across {len(hard_by_pred):,} predecessors "
              f"(mean {hard.groupby('pred_lei').size().mean():.1f}/pred)",
              file=sys.stderr)
    else:
        hard_by_pred = {}

    # ---- build InputExamples ----------------------------------------------
    print(f"building training triples (variant={args.variant})...", file=sys.stderr)
    train_examples = []
    for _, row in pos_train.iterrows():
        pred, succ = row["anchor_lei"], row["candidate_lei"]
        anchor_text = text_for(pred)
        pos_text = text_for(succ)
        negs = hard_by_pred.get(pred, []) if hard_by_pred else []
        if negs and args.variant != "no_hard":
            # pick one hard negative for this triple
            neg = negs[int(rng.integers(0, len(negs)))]
            neg_text = text_for(neg)
            train_examples.append(InputExample(texts=[anchor_text, pos_text, neg_text]))
        else:
            train_examples.append(InputExample(texts=[anchor_text, pos_text]))
    print(f"  {len(train_examples):,} training examples", file=sys.stderr)
    if not train_examples:
        print("no examples — aborting", file=sys.stderr)
        sys.exit(1)

    # ---- train -------------------------------------------------------------
    out_dir = os.path.join(args.data, "models", f"method_{args.variant}")
    os.makedirs(out_dir, exist_ok=True)

    if args.variant == "zero_shot":
        print(f"zero-shot: loading base {args.base_model} and skipping training",
              file=sys.stderr)
        model = SentenceTransformer(args.base_model, device=device)
    else:
        print(f"loading base model {args.base_model}...", file=sys.stderr)
        model = SentenceTransformer(args.base_model, device=device)
        train_dl = DataLoader(train_examples, shuffle=True,
                              batch_size=args.batch_size)
        loss = losses.MultipleNegativesRankingLoss(model)
        print(f"training for {args.epochs} epoch(s), batch={args.batch_size}, "
              f"lr={args.lr}...", file=sys.stderr)
        t0 = time.time()
        model.fit(
            train_objectives=[(train_dl, loss)],
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
            output_path=out_dir,
        )
        print(f"  trained in {time.time()-t0:.1f}s -> saved to {out_dir}",
              file=sys.stderr)

    # ---- score every pair in pairs.parquet --------------------------------
    print("scoring all pairs...", file=sys.stderr)
    leis = list(ent["lei"])
    texts = [text_for(lei) for lei in leis]
    t0 = time.time()
    embs = model.encode(texts, batch_size=128, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
    print(f"  encoded {len(leis):,} entities in {time.time()-t0:.1f}s "
          f"-> {embs.shape}", file=sys.stderr)

    lei_to_idx = {lei: i for i, lei in enumerate(leis)}
    a_idx = pairs["anchor_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    c_idx = pairs["candidate_lei"].map(lei_to_idx).fillna(-1).astype(int).to_numpy()
    valid = (a_idx >= 0) & (c_idx >= 0)
    scores = np.zeros(len(pairs), dtype=np.float32)
    A = embs[a_idx[valid]]
    C = embs[c_idx[valid]]
    scores[valid] = (A * C).sum(axis=1)

    preds_dir = os.path.join(args.data, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"method_{args.variant}.parquet")
    out_df = pd.DataFrame({"anchor_lei": pairs["anchor_lei"],
                            "candidate_lei": pairs["candidate_lei"],
                            "score": scores.astype(float)})
    out_df.to_parquet(out_path, compression="snappy", index=False)
    print(f"wrote -> {out_path}  (mean={scores.mean():.3f}, "
          f"median={float(np.median(scores)):.3f})", file=sys.stderr)

if __name__ == "__main__":
    main()
