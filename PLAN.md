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

## Phase 3 — Method (Weeks 6–8)

### Architecture
- Dual bi-encoder initialized from BGE-large or E5-large-v2.
- Input: `(record_text, jurisdiction, time_state_features)`.
- Time-state features (16-dim projection, no future leakage): listed exchange, HQ country, legal form code, last-known address city, had-lifecycle-event-in-past-365-days flag at time t.

### Training
- InfoNCE with three positive types: identity, event-bridging, successor-chain.
- Hard negatives from Phase 1B.
- Batch 256, ~3 epochs, single GPU.

### Ablations
- No state features.
- No event-bridging positives.
- No hard negatives.

### Budget
- Compute: $0 if personal GPU, ~$200–500 if rented.
- LLM APIs for baselines: ~$500–2000.

## Phase 4 — Eval, writing (Weeks 9–11)

### Week 9
- Full results × baselines × method × splits.
- Per-event-type breakdown.
- Calibration plots.
- Error analysis: 100 sampled failures, categorized.

### Week 10 — Draft
- NeurIPS D&B template, 8–9 pages.
- Sections: Abstract → Intro → Related → Dataset → Method → Experiments → Analysis → Limitations → Conclusion.
- 4 figures, 1 main results table.
- "Why this matters" paragraph: explicit Stripe / Ramp / Bloomberg / Robinhood callouts.

### Week 11 — Polish
- [Datasheet for datasets](https://arxiv.org/abs/1803.09010) (required for D&B track).
- HF Datasets release.
- GitHub README polish.
- Read-through + 1–2 peer reviewers.

## Phase 5 — Submit + outreach (Week 12)

- arXiv preprint: Sep 1, 2026.
- ICAIF 2026 workshop.
- Stretch: NeurIPS 2026 D&B if timing allows.
- Cold outreach: Bloomberg NLP hiring manager, Stripe Radar engineer, Ramp eng recruiting, Robinhood Cortex.
- One X / LinkedIn thread.

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
