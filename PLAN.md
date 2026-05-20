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

### 1A. Core GLEIF benchmark (Week 1)
- [x] Stream-parse Level-1 XML into typed parquet (`src/ingest/gleif_to_parquet.py`). 3.3M entities, 1.17M events, 33.6K successor pairs → 302 MB on disk.
- [x] Enrich successor links via parquet join (`src/benchmark/build_successor_links.py`). 94.57% plausibly real.
- [ ] Download GLEIF Level-2 RR file; write `src/ingest/gleif_rr_to_parquet.py` for relationships.parquet.
- [ ] Build positive pairs from the 33.6K successor links (after dropping likely-administrative). Pull both entities' Level-1 records at multiple time points so each side has temporal context.
- [ ] Three splits:
  - **Held-out-time:** train events <2025-01-01, test ≥2025-01-01.
  - **Held-out-event-type:** train without SPINOFF / DEMERGER / BREAKUP / ABSORPTION, test on those.
  - **Held-out-jurisdiction:** train on US + UK + DE, test on Singapore + Japan + Brazil.

### 1B. Hard-negative mining (Week 2)
- Name-similar negatives (E5-small cosine > 0.85, distinct LEI, no successor link).
- **Parent-subsidiary negatives** from Level 2 RR. (Phase-0 crosswalk surfaced
  the LUXFER INC. / LUXFER HOLDINGS PLC failure mode — these must be in the
  hard-negative set.)
- Same-jurisdiction shell negatives.
- Ratio: 5 negs per pos in train, 10 per pos in test.

### 1C. Supplements (Week 3, optional)
- EDGAR 8-K Item 5.03 name changes (US public-company subset).
- Wikidata high-profile M&A (Twitter → X etc., via P749 / P156 / P155).
- Cross-reference and dedupe against GLEIF labels.

**Deliverable:** `TemporalLink_v1` parquet (~1M labeled pairs) + Hugging Face Datasets entry.

## Phase 2 — Baselines (Weeks 4–5)

| Baseline | Why | Effort |
|---|---|---|
| Exact-match on legal name | Trivial floor | <1 day |
| Dedupe (Fellegi-Sunter) | Classical RL | 1–2 days |
| CompanyName2Vec | Static company-matching baseline | 1 day |
| E5 / BGE / GTE zero-shot | Modern pretrained encoder | 1 day |
| GPT-4o + Claude pairwise (OpenSanctions Pairs style) | LLM SOTA | 2–3 days, ~$500 API |
| DeepSeek-R1-Distill-Qwen-14B local | Open-model LLM | 2 days, needs GPU |

**Gate:** If LLMs already hit >95% on all splits including held-out-event-type, pivot to benchmark-only paper.

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
4. Check NeurIPS 2026 D&B exact deadline
5. Solo vs collaborator decision
6. Start Phase 1A: parse GLEIF Level-1 + Level-2 into parquet
