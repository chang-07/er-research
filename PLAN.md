# Implementation plan

12-week plan, May 19 2026 → Aug 11 2026.

**arXiv preprint target:** Sep 1, 2026 (visible during Summer 2027 internship
application season).

**Primary venue:** ICAIF 2026 workshop (likely Aug deadline, low risk).
**Stretch venue:** NeurIPS 2026 Datasets & Benchmarks track.

## Phase 0 — Pre-flight ✅ DONE

- [x] Audit script committed
- [x] DATA_NOTES.md with headline numbers
- [x] Label-quality validation: **94.57% plausibly real** (gate was 80%). 33,645 resolvable `SuccessorEntity` pairs, 31,817 usable after dropping likely-administrative. See `validate_successor_pairs.py`.
- [x] LEI ↔ CIK crosswalk: **3,334 confident matches** (gate was 1,000). See `lei_cik_crosswalk.py`.

**Gate decision: GO.** Both spikes passed comfortably. Proceeding to Phase 1.

## Phase 1 — Dataset construction (Weeks 1–3)

### 1A. Core GLEIF benchmark ✅ DONE
- [x] Stream-parse Level-1 XML into typed parquet (`src/ingest/gleif_to_parquet.py`). 3.3M entities, 1.17M events, 33.6K successor pairs → 302 MB on disk.
- [x] Enrich successor links via parquet join (`src/benchmark/build_successor_links.py`). 94.57% plausibly real.
- [x] Download + parse GLEIF Level-2 RR (`src/ingest/gleif_rr_to_parquet.py`). 473,839 relationships across 294K children / 76K parents.
- [x] Build positive pairs from the 31.8K plausibly-real successor links.
- [x] Three splits:
  - **Held-out-time:** anchor-event date <2025-01-01 vs. ≥2025-01-01.
  - **Held-out-event-type:** M&A/rename/form-change vs. spinoff/demerger/breakup/absorption.
  - **Held-out-jurisdiction:** US+GB+DE vs. SG+JP+BR.

### 1B. Hard-negative mining ✅ DONE
- [x] Mined 253,733 hard negatives in 4 classes (`src/benchmark/mine_hard_negatives.py`):
  - 232,601 token-overlap same-country (fuzzy fallback)
  - 10,703 same-name different-country
  - 10,208 same-name same-country
  - 221 parent/subsidiary from Level-2 RR
- [x] Ratio applied: up to 5 negs/pos for train-only positives, up to 10 negs/pos for positives that hit any test split.
- [x] Coverage: 63% of positives get at least one hard neg; remaining 37% (small entities in jurisdictions w/o name collisions) fall back to random negatives.
- **v2 improvement (not blocking the paper):** swap token-overlap for proper E5-small cosine similarity; documented in paper limitations.

### 1C. Supplements ✅ partially done
- [x] Wikidata high-profile M&A via SPARQL P156 (`src/benchmark/wikidata_marquee.py`). 302 both-side-matched marquee pairs (Sprint→T-Mobile, Interbrew→AB-InBev, Cannon→MGM, etc.); 246 of them are NEW signal not already in GLEIF.
- [ ] EDGAR 8-K Item 5.03 name changes — deferred; the GLEIF + Wikidata combination already exceeds the v1 size target. Add in v2 if a specific US-public-company analysis section needs the extra coverage.

**Deliverable: TemporalLink v1**
- `pairs.parquet` — 125,519 rows (31,824 positives + 93,695 negatives) with three split assignments
- `wikidata_marquee.parquet` — 3,590 Wikidata bindings, 302 both-side-matched marquee pairs
- All underlying parquet tables (`entities`, `events`, `relationships`, `successor_links`, `hard_negatives`) shipped together
- Full reproduction wall time from raw GLEIF XML: ~7 min

## Phase 2 — Baselines ✅ DONE (autonomous portion)

| Baseline | Run? | Best Hits@1 (time / event / juris) |
|---|---|---|
| exact_name | ✅ | 0.036 / 0.061 / 0.171 |
| token_set (Jaccard) | ✅ | 0.104 / 0.140 / 0.318 |
| tfidf_char (char-3gram) | ✅ | 0.176 / 0.204 / 0.421 |
| sentence_encoder (E5-small-v2 zero-shot) | ✅ | **0.202** / **0.224** / **0.432** |
| llm_pairwise (Claude / GPT) | scaffolded only — user to run | — |
| Dedupe (Fellegi-Sunter) | skipped for v1 — name-feature baselines already saturated | — |

**Gate decision: method paper is justified.** Best name-based baseline tops
out at ~22% Hits@1 on time + event splits; AUROC stays under 0.5 for all
similarity-based baselines (anti-correlated, as expected given the hard-
negative construction). Massive headroom for a lifecycle-aware method.

Full numbers in `DATA_NOTES.md` (Phase 2 section).

LLM run remains: `src/baselines/llm_pairwise.py` is ready; needs API key
(ANTHROPIC_API_KEY or OPENAI_API_KEY) and ~$30 budget for a 5% sample
to get a quick read.

## Phase 3 — Method ✅ DONE (v1)

Trained an E5-small-v2 lifecycle-aware bi-encoder via supervised contrastive
learning on the 26K event-bridging positives (intersection of all three
splits' train sets) plus Phase-1B hard negatives.

| Variant | Hits@1 time / event / juris | Δ vs zero-shot E5 |
|---|:-:|:-:|
| sentence_encoder (zero-shot E5-small-v2) | 0.202 / 0.224 / 0.432 | baseline |
| **method_full** (name + jurisdiction + form + hard negs) | **0.254 / 0.287 / 0.500** | **+5.2 / +6.3 / +6.8 pp** |
| method_no_state (name only) | 0.238 / 0.265 / 0.489 | +3.6 / +4.1 / +5.7 pp |
| **method_no_hard** (in-batch negs only) | **0.258 / 0.295 / 0.511** | **+5.6 / +7.1 / +7.9 pp** |

Surprise: `no_hard` slightly beats `full`. The `token_overlap_same_country`
hard-neg class (Phase-1B fuzzy fallback) is noisy; clean E5-cosine-mined
hard negs in v2 should reverse this.

Training: 6 minutes per variant on Apple Silicon MPS. No GPU required for v1.

State features ablation contributes ~1.5–2 pp Hits@1 — small but real.

Full results table in `DATA_NOTES.md` (Phase 3 section).

## Phase 4 — Eval, writing ✅ DONE (v1 draft)

- [x] Extended eval: per-event-type Hits@1, error analysis on 100 method failures, event-year histogram (`src/eval/extended.py`).
- [x] Figures: event-year stacked bar, main results bar chart, per-event-type breakdown (`src/eval/figures.py` → `paper/figures/*.pdf`).
- [x] Paper draft: `paper/draft.tex` + `paper/refs.bib`. ~9 pages: abstract through conclusion. Article-class placeholder; swap to venue style at submission.
- [x] Datasheet: `paper/datasheet.md` (Gebru et al. template). Required appendix for NeurIPS D&B and ACL data tracks.
- [x] `paper/README.md` with compile + submission checklist.
- [ ] HF Datasets release (still v1.1 work — needs HF account + dataset card).
- [ ] Read-through + 1-2 peer reviewers (your call).

Phase 4 v1 is complete. The remaining items (HF release, peer review) are
about submission logistics rather than research content.

## Phase v2 — integration ✅ DONE

Implemented the three v2 items from the original future-work list, plus a
fourth one (cross-encoder) that turned into the headline result:

- [x] **v2.1 E5-cosine hard-neg re-mining** (`src/benchmark/mine_hard_negatives_v2.py`). 631K E5-cosine same-country negatives replace the noisy token-overlap fallback; coverage rises from ~20K to all 31,714 plausibly-real predecessors (mean 20.6 negs each).
- [x] **v2.2 Transitive successor positives** (`src/benchmark/transitive_positives.py`). 1,372 chained pairs from depths 2–4. Now emits `intermediate_leis` for path-aware leakage filtering downstream.
- [x] **v2.3 Bi-encoder retrain** (`method_v2_full`). Hits@1 0.249 / 0.292 / 0.500 — comparable to v1 best. Erases the v1 "hard negs hurt" finding; in-batch alone is now tied with explicit-hard-neg training.
- [x] **v2.4 Cross-encoder fine-tune** (`src/method/train_cross_encoder.py`). The headline v2 result:

| Split | Hits@1 | AUROC | F1 |
|---|:-:|:-:|:-:|
| time | **0.605** (+34.7 pp vs v1) | **0.836** | **0.61** |
| event | **0.564** (+26.9 pp vs v1) | **0.825** | **0.58** |
| jurisdiction | **0.750** (+23.9 pp vs v1) | **0.868** | **0.67** |

First method to lift AUROC above 0.5 on the time/event splits — the
anti-correlation regime is structurally a bi-encoder problem; joint
encoding escapes it.

Full v2 numbers + ablation are documented in DATA_NOTES.md "v2 integration"
section; paper draft has been updated to reflect both methods.

## Phase v3 — miner robustness ✅ DONE

The Tier-1 paper improvement identified during the goal-driven audit:
test whether the cross-encoder gain depends on the choice of hard-negative
miner used to build the evaluation candidate pool.

- [x] **v3.1 TF-IDF char-3gram miner** (`src/benchmark/mine_hard_negatives_v3.py`). Structurally orthogonal to v2's E5 cosine: no learned representation, no shared parameters with the encoder family. 631K alternative negs.
- [x] **v3.2 Parallel pairs file** (`data/pairs_v3.parquet`, 217K rows) built via parameterized `build_splits.py --hard-negs hard_negatives_v3.parquet --out-name pairs_v3.parquet`.
- [x] **v3.3 Second cross-encoder** trained on v3 hard negs (`method_v3_crossenc`); both v2- and v3-trained cross-encoders scored on both candidate pools, producing a 2x2 train/eval matrix.
- [x] **v3.4 Comparison + bootstrap CIs** via `src/eval/compare_distributions.py` and `src/eval/compute_robustness_cis.py`. Key result:

| Train \ Eval | v2 pool TIME | v2 pool EVENT | v2 pool JURIS | v3 pool TIME | v3 pool EVENT | v3 pool JURIS |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| cross-enc trn-on-v2 (repro) | 0.600 ±2.2 | 0.551 ±2.4 | 0.705 ±8.8 | 0.545 ±1.8 | 0.528 ±1.7 | 0.571 ±7.5 |
| cross-enc trn-on-v3 | 0.588 ±2.0 | 0.533 ±2.4 | 0.750 ±9.1 | **0.655** ±1.5 | **0.614** ±1.7 | 0.640 ±7.1 |
| E5-small zero-shot | 0.203 | 0.234 | 0.432 | 0.265 | 0.319 | 0.441 |

Cross-encoder gain over the baseline: $+20$ to $+40$ pp on in-distribution cells (diagonal), $+13$ to $+38$ pp on out-of-distribution cells. Gap is partly in-distribution but not purely a within-miner artifact.

- [x] **v3.5 Bootstrap CIs** added to `src/eval/evaluate.py` for the main results table — jurisdiction split's headline now reported with $\pm$8.8 pp CI.
- [x] **v3.6 Per-event-type breakdown on the event split** added via `src/eval/extended.py` — surfaces that the "held-out event type" test set is 94.7% ABSORPTION (DEMERGER, BREAKUP, SPINOFF contribute n=42, 20, 4).
- [x] **v3.7 Path-aware transitive positive filter** in `train.py` and `train_cross_encoder.py` rejects transitive (A, C) pairs whose chain crosses a train/test boundary.
- [x] **v3.8 Paper updates:** miner-robustness section (Sec~\ref{ssec:robustness}), CI ranges in Table 1 caption, honest re-framing of anti-correlation finding as construction artifact, expanded limitations section (8 paragraphs).

## Phase v3.5 - pre-ICAIF hardening (in progress)

Driven by reviewer-attack-pattern research (D&B chair commentary, OpenReview
patterns) and 2024-2026 ER benchmark conventions (Peeters/EDBT, ComEM,
OpenSanctions Pairs). Goal: address the top reviewer concerns before ICAIF
2026 submission (Aug 2 deadline).

- [x] **v3.5.1 Cost & latency table** per method (`src/eval/cost_latency.py`). 8 LLM rows priced at late-2025 list with system-prompt caching.
- [x] **v3.5.2 Calibration analysis** (`src/eval/calibration.py`): reliability diagram + ECE/Brier. Cross-encoder ECE=0.203, suggests downstream temperature scaling.
- [x] **v3.5.3 Per-jurisdiction breakdown** on the held-out-jurisdiction split: SG/JP/BR separately. Cross-encoder wins +29/+55/+29 pp over E5 zero-shot per country.
- [x] **v3.5.4 Cluster-level metrics (B-cubed)** (`src/eval/cluster_metrics.py`). Cross-encoder F1 0.90-0.92 vs bi-encoder ≤0.47; cosine-based clustering collapses everything into a giant cluster at the tuned threshold.
- [x] **v3.5.5 Named error taxonomy** on 122 sampled cross-encoder failures (`src/eval/error_taxonomy.py`). 76% are same-country lexical-overlap distractors; 7% are no-shared-tokens cases (the genuinely hard residual).
- [x] **v3.5.6 Contamination statement** for LLM baselines (GLEIF is public; frontier LLMs may have memorized).
- [x] **v3.5.7 Secondary label-quality check** (`src/eval/secondary_label_check.py`): orthogonal heuristic re-classifier on 277 sampled positives agrees with primary at 98.7% (95% CI 97.0-100%), bounding label noise <2% on confident-positive subset.
- [x] **v3.5.8 Croissant 1.0 metadata** (`data/croissant.json`) + HF dataset card (`data/HF_README.md`). NeurIPS D&B desk-reject criterion as of 2025.
- [x] **v3.5.9 Benchmark comparison table** vs TempEL, OpenSanctions Pairs, Magellan, WDC Products, Ditto-WDC.
- [x] **v3.5.10 Cross-dataset transfer** to Magellan Walmart-Amazon (`src/method/score_external.py`, `src/method/score_external_e5.py`). Honest negative result: our CE underperforms zero-shot E5 by 5.9pp AUROC, indicating domain specialization (not generic ER transferability).
- [x] **v3.5.11 Larger bi-encoder backbone** (E5-base-v2, 110M params, 3x). TIME -0.2pp, EVENT -1.2pp, JURIS +2.3pp Hits@1 — bi-encoder ceiling is structural, not under-parameterized.
- [ ] **v3.5.12 Larger cross-encoder backbone** (ms-marco-MiniLM-L-12-v2, 60M params, 3x) — training in progress.
- [ ] **v3.5.13 Data-efficiency scaling curve** (`src/method/run_data_efficiency.sh`): bi-encoder retrains at 10/25/50/75/100% training data — pending.
- [x] **v3.5.14 Format swap to ACM sigconf** for ICAIF 2026 — preamble done; aggressive cut to 8 pages pending.
- [ ] **v3.5.15 LLM panel baseline** (3-LLM minimum per Peeters/EDBT 2025 conventions: Haiku 4.5 + GPT-4o-mini + Llama-3.1-8B). User has API credits; script ready.

## Phase 5 - Submit + outreach (revised after May 2026 venue research)

NeurIPS 2026 D&B deadline was May 6, 2026 — missed. Primary target shifts to ICAIF 2026 (Aug 2, 2026, ACM SIGCONF, 8 pages).

- arXiv preprint: Sep 1, 2026 (post-ICAIF submission).
- **Primary: ICAIF 2026 main track**, Milan, deadline Aug 2, 2026. 8-page ACM sigconf. Good fit alongside FinGraphEx/FinReflectKG/FinDER (ICAIF 2025).
- Parallel non-archival: NeurIPS 2026 GenAI in Finance workshop (deadline TBA ~Aug-Sep 2026, announced July 11).
- Backup: ECIR 2027 Resource Track (Nov 2, 2026 deadline; LNCS ~12 pages).
- Cold outreach: Bloomberg NLP hiring manager, Stripe Radar engineer, Ramp eng recruiting, Robinhood Cortex.
- One X / LinkedIn thread when arXiv is live.

## Risks

| Risk | Mitigation |
|---|---|
| Label quality lower than expected | Phase 0 validation gate; filter to `COMPLETED` + supporting docs |
| LLMs saturate the task | Benchmark-only paper still publishable at D&B; held-out-event-type is the insurance |
| Method doesn't beat baselines | Same as above |
| GPU access | Colab Pro+ or Lambda spot ($1–2/hr H100) |
| Scoop in next 3 months | Ship arXiv as soon as baselines + dataset done (Week 5), don't wait for method |
| Venue deadlines slip | arXiv is the real deadline |

## This week

1. ✅ Audit pushed to git
2. ✅ Validation pass: 94.57% plausibly real
3. ✅ Crosswalk spike: 3,334 confident matches
4. ✅ Phase 1A: GLEIF Level-1 + Level-2 ingested to parquet
5. ✅ Phase 1B: hard negatives mined; three eval splits assembled
6. ✅ Phase 1C: Wikidata marquee supplement
7. Check NeurIPS 2026 D&B exact deadline
8. Solo vs collaborator decision
9. **Start Phase 2: baselines** (Dedupe, sentence encoders, LLM pairwise)
