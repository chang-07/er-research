# TemporalLink

**Lifecycle-aware entity resolution for financial entities.**

Most entity-resolution systems answer "do these two records refer to the same
entity *right now*?" They fall apart when the entity has undergone a lifecycle
event — a merger, spinoff, rebrand, ticker change, jurisdiction migration —
between the two records. This repo builds the first open benchmark for that
problem and a method that beats both classical record-linkage and modern
LLM-based pair-matching baselines on lifecycle-spanning pairs.

All data is from open registries (GLEIF, SEC EDGAR, Wikidata). No proprietary
sources required.

## Status

- [x] Data-availability audit on the GLEIF Level-1 LEI-CDF dump (`gleif_audit.py`)
- [x] Label-quality validation: 94.57% plausibly real (`validate_successor_pairs.py`)
- [x] LEI ↔ CIK crosswalk: 3,334 confident matches (`lei_cik_crosswalk.py`)
- [x] Parquet ingestion: 3.3M entities, 1.17M events, 33.6K successor pairs (`src/ingest/gleif_to_parquet.py` + `src/benchmark/build_successor_links.py`)
- [x] GLEIF Level-2 RR ingestion: 473,839 relationships (`src/ingest/gleif_rr_to_parquet.py`)
- [x] Hard-negative mining: 253,733 candidates across 4 classes (`src/benchmark/mine_hard_negatives.py`)
- [x] Eval splits built: 125,519 pair rows with held-out-time / -event-type / -jurisdiction assignments (`src/benchmark/build_splits.py`)
- [x] Wikidata marquee supplement: 302 famous M&A matched to GLEIF (`src/benchmark/wikidata_marquee.py`)
- [x] Baselines run: exact_name, token_set, tfidf_char, sentence_encoder (E5-small). Best Hits@1 = 22% on the headline splits — method paper justified. `src/baselines/`, `src/eval/`
- [ ] LLM-pairwise baseline (scaffold ready; needs API key + budget)
- [x] Method (v1): lifecycle-aware bi-encoder fine-tuned on E5-small with state features + hard negatives. **+5–8 pp Hits@1 over zero-shot E5 across all splits.** Two ablations run. `src/method/train.py`
- [x] Paper draft + datasheet + figures (Phase 4): `paper/draft.tex`, `paper/datasheet.md`, `paper/figures/*.pdf`. ~9 pages, Gebru-template datasheet appendix.
- [x] **v2 integration:** E5-cosine hard-neg re-mining (`src/benchmark/mine_hard_negatives_v2.py`), transitive successor positives (`src/benchmark/transitive_positives.py`), v2 bi-encoder retrain, and a fine-tuned cross-encoder (`src/method/train_cross_encoder.py`). **Cross-encoder lifts Hits@1 to 0.605 / 0.564 / 0.750 on the three splits — +24-35 pp over the best bi-encoder, and the first method to break AUROC > 0.5 on the time/event splits.**
- [x] **v3 miner-robustness experiment:** alternative hard-neg pool from char-3gram TF-IDF cosine (`src/benchmark/mine_hard_negatives_v3.py`), parallel pairs file (`data/pairs_v3.parquet`), cross-encoders trained and evaluated in a 2x2 train/eval matrix (`src/method/train_cross_encoder.py` + `src/eval/compare_distributions.py`). **Cross-encoder gain shrinks 5-15 pp under out-of-distribution candidate-pool shift but stays +13 to +38 pp above baselines across all six cells — gap is partly in-distribution but not purely a within-miner artifact.**
- [x] **Statistical rigor:** bootstrap 95% CIs on Hits@1 added to `src/eval/evaluate.py`; jurisdiction split's headline number (n=88 anchors) reported with ±8.8 pp CI band so it isn't over-interpreted.
- [x] **Path-aware transitive-positive filter:** `transitive_positives.py` now emits `intermediate_leis`; both `train.py` and `train_cross_encoder.py` reject transitive (A,C) positives whose chain crosses a train/test boundary.
- [ ] HF Datasets release (manual upload after paper submission).
- [ ] LLM-pairwise baseline (~$2 for 5% Haiku sample; biggest remaining experimental gap)
- [ ] Method v3: explicit modeling of parent/subsidiary edges (the one negative class the cross-encoder regressed on); possibly bigger backbones (E5-large / BGE-large) with GPU.
- [ ] Method (entity-state-at-time bi-encoder w/ event-bridging contrastive loss)
- [ ] Write-up + arXiv preprint
- [ ] Submit (target: ICAIF 2026 workshop primary, NeurIPS 2026 D&B stretch)

See [PLAN.md](PLAN.md) for the full 12-week plan.
See [DATA_NOTES.md](DATA_NOTES.md) for what the GLEIF audit told us.

## Reproducing the Phase-0 spikes

You need the latest GLEIF Level-1 concatenated file (XML, ~2 GB compressed,
~7.5 GB decompressed). Download from
<https://www.gleif.org/en/lei-data/gleif-concatenated-file/download-the-concatenated-file>
and place it at the project root. Then:

```bash
pip install -r requirements.txt
python3 gleif_audit.py              # ~50s — viability audit
python3 validate_successor_pairs.py # ~90s — label-quality check on successor pairs
python3 lei_cik_crosswalk.py        # ~50s + 1 small SEC fetch
```

Note: the SEC fetch in `lei_cik_crosswalk.py` requires an email-bearing
User-Agent. Edit the `UA` constant before running with your own contact.

## Building TemporalLink v1 (all of Phase 1)

```bash
# 1A — typed corpus from GLEIF Level-1 + Level-2 XML
python3 src/ingest/gleif_to_parquet.py    --xml <Level1.xml> --out data/   # ~160s
python3 src/ingest/gleif_rr_to_parquet.py --xml <RR.xml>     --out data/   # ~10s
python3 src/benchmark/build_successor_links.py --data data/                # ~3s

# 1B — hard negatives + eval splits
python3 src/benchmark/mine_hard_negatives.py --data data/                  # ~5 min
python3 src/benchmark/build_splits.py        --data data/                  # ~10s

# 1C — Wikidata marquee supplement
python3 src/benchmark/wikidata_marquee.py    --data data/                  # ~30s
```

Total ~7 minutes from raw XML to `data/pairs.parquet` (125K labeled pair
rows with three split regimes) + `data/wikidata_marquee.parquet`
(302 famous-case matches).

The XML inputs come from
<https://goldencopy.gleif.org/> (Level-1 LEI2 + Level-2 RR
golden-copy files, published every 8 hours, all CC0).

## Running baselines (Phase 2)

```bash
python3 src/baselines/cheap.py   --data data/   # ~1s  — exact_name, token_set, tfidf_char
python3 src/baselines/encoder.py --data data/   # ~40s — E5-small-v2 zero-shot
python3 src/eval/evaluate.py     --data data/   # ~7s  — AUROC, AUPRC, Hits@K, MRR, per-class breakdown
```

LLM pairwise baseline (Anthropic / OpenAI) is scaffolded but not run
autonomously — `src/baselines/llm_pairwise.py` needs an API key and budget.
Use `--sample-frac 0.05` first for a cheap signal before paying for a full
sweep.

## Training the method (Phase 3)

```bash
python3 src/method/train.py --data data --variant full      # ~7 min on Apple MPS
python3 src/method/train.py --data data --variant no_state  # ablation
python3 src/method/train.py --data data --variant no_hard   # ablation
python3 src/eval/evaluate.py --data data                    # combined eval
```

Each variant trains for 3 epochs at batch 32, learning rate 2e-5, and writes
`data/predictions/method_<variant>.parquet` plus a saved checkpoint under
`data/models/method_<variant>/`. The default training set is the
26K positives that fall in `train` for *all three* split regimes — strict no-
leakage. Default device is Apple MPS if available, else CUDA, else CPU.

## v2 pipeline (in addition to Phase 1-3)

```bash
# v2 hard-neg re-mining via E5 cosine (~7 min on Apple MPS)
python3 src/benchmark/mine_hard_negatives_v2.py --data data/

# transitive successor-chain positives (~3 s, includes intermediate_leis
# for path-aware leakage filtering downstream)
python3 src/benchmark/transitive_positives.py --data data/

# v2 bi-encoder retrain (~7 min)
python3 src/method/train.py --variant v2_full --data data/

# v2 cross-encoder fine-tune (~10 min train + 7 min scoring on MPS)
python3 src/method/train_cross_encoder.py --data data/

# eval (with bootstrap 95% CIs on Hits@1)
python3 src/eval/evaluate.py --data data/
```

## v3 miner-robustness experiment

Tests whether the cross-encoder's headline gain depends on the choice of
hard-negative miner used to build the evaluation candidate pool. v3 uses
char-3gram TF-IDF cosine (no learned representation, no shared parameters
with E5) to mine an alternative pool, then trains a second cross-encoder
on it. Both cross-encoders are scored on both pools, yielding a 2x2
train/eval matrix.

```bash
# 1. mine alternative hard negatives (TF-IDF char-3gram, ~20 s CPU)
python3 src/benchmark/mine_hard_negatives_v3.py --data data/

# 2. assemble alternative pairs file with the new negs
python3 src/benchmark/build_splits.py --data data/ \
    --hard-negs hard_negatives_v3.parquet --out-name pairs_v3.parquet

# 3. score the existing baselines on the v3 pool
python3 src/baselines/cheap.py --data data/ \
    --pairs pairs_v3.parquet --suffix _v3
python3 src/baselines/encoder.py --data data/ \
    --pairs pairs_v3.parquet --name sentence_encoder_v3

# 4. retrain the cross-encoder on v2 negs and score on both pools (~17 min)
python3 src/method/train_cross_encoder.py --data data/ \
    --hard-negs hard_negatives_v2.parquet --train-pairs pairs.parquet \
    --extra-pairs pairs_v3.parquet --out-name method_v2_crossenc_repro

# 5. train a second cross-encoder on v3 negs and score on both pools (~20 min)
python3 src/method/train_cross_encoder.py --data data/ \
    --hard-negs hard_negatives_v3.parquet --train-pairs pairs_v3.parquet \
    --extra-pairs pairs.parquet --out-name method_v3_crossenc

# 6. compute and print the comparison + bootstrap CIs
python3 src/eval/compare_distributions.py --data data/ --skip-ci
python3 src/eval/compute_robustness_cis.py
```

## Additional diagnostics (Phase v3.5)

```bash
# Cost & latency table per method (CPU timing + LLM list prices)
python3 src/eval/cost_latency.py

# Reliability diagram + ECE/Brier on cross-encoder
python3 src/eval/calibration.py

# B-cubed cluster precision/recall (Binette 2024 style)
python3 src/eval/cluster_metrics.py

# Named error taxonomy on cross-encoder failures (50 per split)
python3 src/eval/error_taxonomy.py

# Secondary auto-classifier label-quality check (label-noise lower bound)
python3 src/eval/human_validation.py --data data --n 300        # sample
python3 src/eval/secondary_label_check.py                        # label heuristically
python3 src/eval/human_validation.py --data data --report        # report agreement

# Cross-dataset transfer (run trained CE on Magellan Walmart-Amazon)
python3 src/method/score_external.py \
    --model-dir data/models/method_v2_crossenc_saved \
    --test-file /tmp/magellan/walmart_amazon_test.txt \
    --train-file /tmp/magellan/walmart_amazon_train.txt \
    --out data/predictions/method_v2_crossenc_on_walmart_amazon.parquet

# Larger backbones (E5-base bi-encoder + MiniLM-L-12 cross-encoder)
python3 src/method/train.py --data data --variant v2_full \
    --base-model intfloat/e5-base-v2 --out-name method_v2_full_e5base
python3 src/method/train_cross_encoder.py --data data \
    --hard-negs hard_negatives_v2.parquet --train-pairs pairs.parquet \
    --base-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --out-name method_v2_crossenc_L12

# Data-efficiency scaling curve (bi-encoder retrains at 10/25/50/75/100% data)
bash src/method/run_data_efficiency.sh
python3 src/eval/data_efficiency.py

# One-shot wrapper that runs cross-dataset transfer + L-12 + data-efficiency
# in sequence after a saved v2 cross-encoder checkpoint exists
bash src/method/chained_compute.sh
```

## Paper draft

`paper/draft.tex` is a ~9-page LaTeX draft of the TemporalLink paper, with
the full experimental story (bi-encoder + cross-encoder methods, three
figures, Gebru-style datasheet at `paper/datasheet.md`). The draft uses
the standard `article` class — swap to the venue style file at submission
time. See `paper/README.md` for compile instructions.

## License

TBD — most likely MIT for code, CC-BY-4.0 for any derived datasets we release
(GLEIF data is itself CC0).
