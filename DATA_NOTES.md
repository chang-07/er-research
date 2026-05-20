# DATA_NOTES

Persistent record of empirical facts about the GLEIF data so we don't have to
re-derive them. Update whenever a fact changes.

## Source dump

- File: `20260519-gleif-concatenated-file-lei2.xml`
- Snapshot date: **2026-05-19**
- Header `RecordCount`: **3,313,039** (matches parsed total)
- XML namespace: `http://www.gleif.org/data/schema/leidata/2016`
- Decompressed size: ~7.5 GB

## Headline counts (from `gleif_audit.py`, run 2026-05-19)

| Quantity | Count | % of total |
|---|---:|---:|
| Total entities | 3,313,039 | 100.000% |
| Entities with `SuccessorEntity` link | **50,202** | 1.515% |
| Entities with any `LegalEntityEvents` | 662,717 | 20.003% |
| Entities with ≥1 `COMPLETED` event | 656,031 | 19.801% |
| Total `LegalEntityEvent` records | 1,170,716 | — |

## Event-type breakdown (all statuses)

| Type | Count | % of events |
|---|---:|---:|
| CHANGE_LEGAL_ADDRESS | 365,675 | 31.24% |
| CHANGE_HQ_ADDRESS | 355,484 | 30.36% |
| DISSOLUTION | 188,567 | 16.11% |
| CHANGE_LEGAL_NAME | 116,350 | 9.94% |
| CHANGE_LEGAL_FORM | 45,894 | 3.92% |
| MERGERS_AND_ACQUISITIONS | 42,915 | 3.67% |
| LIQUIDATION | 25,881 | 2.21% |
| CHANGE_OTHER_NAMES | 19,289 | 1.65% |
| ABSORPTION | 6,852 | 0.59% |
| BANKRUPTCY | 2,211 | 0.19% |
| DEMERGER | 758 | 0.06% |
| INSOLVENCY | 398 | 0.03% |
| VOLUNTARY_ARRANGEMENT | 258 | 0.02% |
| BREAKUP | 124 | 0.01% |
| SPINOFF | 33 | 0.00% |
| ACQUISITION_BRANCH | 11 | 0.00% |
| TRANSFORMATION_UMBRELLA_TO_STANDALONE | 9 | 0.00% |
| TRANSFORMATION_BRANCH_TO_SUBSIDIARY | 4 | 0.00% |
| TRANSFORMATION_SUBSIDIARY_TO_BRANCH | 3 | 0.00% |

## Status filter is essentially free

97%+ of high-signal event types are already `COMPLETED`. Filtering to
`event_status="COMPLETED"` removes label noise without meaningfully shrinking
the dataset.

## The 2022 cliff

Event volume is small pre-2022 and ~5× larger from 2022 onward. GLEIF
appears to have tightened event-reporting requirements at that time. Pre-2022
data exists but is sparse and biased.

| Year | Total events |
|---|---:|
| 2014 | 5,131 |
| 2015 | 8,119 |
| 2016 | 10,474 |
| 2017 | 11,676 |
| 2018 | 18,528 |
| 2019 | 24,826 |
| 2020 | 28,640 |
| 2021 | 44,264 |
| **2022** | **184,052** |
| 2023 | 209,835 |
| 2024 | 261,622 |
| 2025 | 260,771 |
| 2026 (YTD May) | 86,275 |

**Implication for splits:** main eval window is 2022–2026. Pre-2022 is held
out as a sparser out-of-distribution test.

## What's signal vs noise for our task

| Category | Count | Use |
|---|---:|---|
| Real lifecycle events (M&A + rename + form change + spinoff + demerger + absorption + breakup) | ~401,493 | **positive labels** |
| Of which COMPLETED | ~392,889 | usable directly |
| Address changes (legal + HQ) | ~721,159 | **hard negatives** (same entity, irrelevant change) |
| Dissolutions / liquidations / bankruptcies | ~216,659 | exit events; useful for entity-death modeling, not main signal |
| Direct `SuccessorEntity` links | **50,202** | 🥇 **gold-standard predecessor → successor pairs** — direct supervision, no annotation |

## Sample successor links (sanity check)

These appear to be real M&A events, not administrative re-issuances:

- STANBIC IBTC BANK PLC → 549300NIVXF92ZIOVW61
- UNITY BANK PLC → 0292003440E3M3A4D637
- DIAMOND BANK PLC → 029200328C3N9YI2D660
- AXA MANSARD INSURANCE PLC → 029200324B5N5B01F690
- CUTIX PLC → 029200894A3T8E0D9I44

(File appears partially sorted by LEI prefix; the first samples skew Nigerian
because of LOU prefix ordering, not because of geographic bias in the data
itself.)

## Verdict

**GO.** High-signal event count (~401K, ~393K COMPLETED) is comfortably 20× the
20K threshold I set as the paper viability floor.

---

# Phase-0 spike results (2026-05-19)

## Successor-pair label quality (`validate_successor_pairs.py`)

For each `SuccessorEntity` link, fetched both predecessor and successor Level-1
records and classified the pair.

| Quantity | Count |
|---|---:|
| Entities with `SuccessorEntity` element (audit) | 50,202 |
| Pairs with a populated `SuccessorLEI` (resolvable) | **33,645** |
| Successors actually present in this snapshot | 26,849 (100% of resolvable) |

The gap of ~16,557 between "has SuccessorEntity element" and "has resolvable
SuccessorLEI" is entities whose successor never registered an LEI (e.g.,
acquired into a private parent that didn't register, or dissolved into a
non-LEI entity). These are still usable for entity-death modeling and for
matching to non-GLEIF sources (EDGAR, Wikidata).

**Classification of the 33,645 resolvable pairs:**

| Class | Count | % |
|---|---:|---:|
| strong_real (documented M&A/rename event + name actually changed) | 25,428 | 75.58% |
| likely_real_name_changed (name differs, no event flag) | 3,330 | 9.90% |
| likely_real_event_only (event documented but name unchanged) | 1,704 | 5.06% |
| ambiguous | 1,355 | 4.03% |
| likely_administrative (everything identical) | 1,828 | **5.43%** |

**Plausibly-real rate: 94.57%.** Well above the 80% viability gate.
Filter strategy: drop the 5.43% `likely_administrative` (same name, same
country, same legal form, no documented event) and keep everything else.

100 random pairs dumped to `data/successor_validation_sample.csv` for
spot-check (gitignored).

## LEI ↔ CIK crosswalk feasibility (`lei_cik_crosswalk.py`)

Fetched SEC's `company_tickers.json` (10,354 records), normalized names, and
hash-joined against GLEIF's US-jurisdiction entities.

| Quantity | Count |
|---|---:|
| GLEIF entities with US jurisdiction | 351,018 |
| SEC ticker records | 10,354 |
| Unique normalized SEC names | 7,967 (collision rate 23%) |
| **GLEIF → single CIK (confident link)** | **3,334** |
| GLEIF → multiple CIKs (ambiguous) | 564 |
| GLEIF, no match | 347,120 |
| % of SEC filers found in GLEIF (US) | ~41.8% |

**Spike target was ≥1,000 confident matches. Hit 3,334. PASS.**

The 58% of SEC filers we don't find in US-GLEIF mostly reflects SEC filers
incorporated in Cayman, Delaware-with-foreign-parent, or otherwise registered
elsewhere — i.e., they exist in GLEIF under a different jurisdiction. A full
crosswalk should not restrict GLEIF to `country=US` but instead match across
jurisdictions.

**Two methodological observations from the sample matches** (both should
influence Phase-1 hard-negative mining):

1. **Parent-subsidiary false matches.** `LUXFER INC.` (US LLC) normalized to
   the same name as `LUXFER HOLDINGS PLC` (UK parent CIK). Similar for
   FORTREA INC. → Fortrea Holdings Inc. The paper's hard-negative set
   should explicitly include parent/subsidiary pairs that look similar but
   are legally distinct entities.
2. **Ticker-variant ambiguity.** Most "1 name → multiple CIKs" cases are the
   same CIK with different ticker symbols (e.g., `SKILSOFT CORPORATION` →
   SKIL + SKILW for common stock + warrants). These are not entity-resolution
   problems per se; deduping on CIK (not ticker) before joining resolves them.

## What changes in PLAN.md as a result

- Phase 0 gates **passed**. Move to Phase 1.
- In Phase 1B (hard negatives), explicitly include parent-subsidiary pairs
  from the LUXFER-style failure mode.
- In Phase 1A (positive labels), filter out `likely_administrative` cases
  (~5.4% of raw successor pairs) — yields a clean ~31,800-pair core training
  set from GLEIF alone, before EDGAR/Wikidata supplements.
- In Phase 1C (EDGAR supplement), the 3,334 confident LEI↔CIK matches are
  enough to bootstrap 8-K Item 5.03 enrichment for the US public-company
  slice. Add OpenFIGI fallback later if we need to widen coverage.

---

# Phase 1A: parquet ingestion (2026-05-19)

`src/ingest/gleif_to_parquet.py` performs a single streaming pass over the
GLEIF Level-1 XML and emits three parquet tables.
`src/benchmark/build_successor_links.py` then joins them to produce the
enriched + classified successor-link table.

## Outputs

| File | Rows | Size (snappy) |
|---|---:|---:|
| `data/entities.parquet` | 3,313,039 | 275.5 MB |
| `data/events.parquet` | 1,170,716 | 22.7 MB |
| `data/successor_links_raw.parquet` | 33,645 | 1.1 MB |
| `data/successor_links.parquet` (enriched) | 33,652 | 2.6 MB |

Total: ~302 MB on disk for the full Level-1 corpus, a ~27× compression vs.
the 7.5 GB source XML. All downstream code reads from parquet; no further
need to touch the XML.

The 7-row difference between raw (33,645) and enriched (33,652) successor
tables is from the entities-side merge: a small number of predecessor LEIs
have successor LEIs that appear more than once in the entities table
(GLEIF historical re-issuances). Trivial; will be deduped in Phase 1B.

## Runtime

- Ingestion: 157.8 s wall (21K records/s steady-state) on M-series MacBook.
- Successor enrichment: ~3 s including parquet I/O.
- Pure-Python with `lxml.iterparse` streaming; constant memory.

## Cross-table sanity checks

| Query | Value | Cross-reference |
|---|---:|---|
| Total entities | 3,313,039 | Matches GLEIF XML header `RecordCount` |
| ACTIVE entities | 3,064,538 (92.5%) | Sensible |
| US-jurisdiction entities | 351,018 | Matches Phase 0 crosswalk |
| All MERGERS_AND_ACQUISITIONS events | 42,915 | Matches audit |
| COMPLETED M&A events | 41,708 | Matches audit |
| COMPLETED M&A in 2023 | 4,143 | Plausible |
| US → non-US successor migrations | 28 | Small but non-zero (cross-border reorgs) |

## Schemas (high level)

`entities.parquet`:
`lei, legal_name, other_names[], legal_form_code, legal_jurisdiction,
category, sub_category, entity_status, entity_creation_date,
entity_expiration_date, entity_expiration_reason,
legal_addr_{country,city,region,postal_code}, hq_addr_{country,city,region,postal_code},
associated_entity_lei, initial_registration_date, last_update_date,
next_renewal_date, registration_status, validation_sources, managing_lou,
has_successor, successor_lei`

`events.parquet`:
`lei, event_type, event_status, group_type, effective_date, recorded_date,
validation_documents, affected_field_xpaths, affected_field_values`

`successor_links.parquet`:
`pred_lei, succ_lei, pred_name, succ_name, pred_country, succ_country,
pred_form, succ_form, pred_status, succ_status, has_real_event,
n_real_events, classification`

---

# Phase 1A.2 + 1B + 1C: rest of Phase 1 (2026-05-19)

## Level-2 RR ingestion (1A continued)

Downloaded the latest Level-2 RR golden-copy XML (892 MB decompressed) and
parsed it via `src/ingest/gleif_rr_to_parquet.py`.

| Quantity | Value |
|---|---:|
| Relationship records | 473,839 |
| ACTIVE relationships | 473,419 (99.9%) |
| Unique children (start_lei) | 294,548 |
| Unique parents (end_lei) | 75,864 |
| Output `relationships.parquet` | 10.1 MB |
| Runtime | 10.4 s (46K rec/s) |

Relationship-type mix:

| Type | Count |
|---|---:|
| IS_FUND-MANAGED_BY | 145,666 |
| IS_ULTIMATELY_CONSOLIDATED_BY | 129,772 |
| IS_DIRECTLY_CONSOLIDATED_BY | 123,967 |
| IS_SUBFUND_OF | 71,195 |
| IS_INTERNATIONAL_BRANCH_OF | 1,904 |
| IS_FEEDER_TO | 1,335 |

## Hard-negative mining (1B)

`src/benchmark/mine_hard_negatives.py` mines four classes of hard negatives
per real positive:

| Class | Description | Count (deduped) |
|---|---|---:|
| token_overlap_same_country | ≥2 shared content tokens, same country, fuzzy fallback | 232,601 |
| same_name_diff_country | Identical normalized name in different country | 10,703 |
| same_name_same_country | Identical normalized name in same country | 10,208 |
| parent_subsidiary | Parent/subsidiary edge from Level-2 RR | 221 |
| **Total** | | **253,733** |

Coverage: ~20,138 out of 31,817 positive predecessors (~63%) get at least one
hard neg. The remaining ~37% are small entities in jurisdictions without
name collisions — they fall back to random negatives during training.

Two findings of methodological interest:

1. The `parent_subsidiary` class is small (221) because predecessors of M&A
   events typically lose their active parent/subsidiary relationships at the
   time of acquisition. RR captures *current* corporate hierarchy, not the
   pre-event one.
2. The fuzzy fallback (`token_overlap_same_country`) dominates volume but
   has the lowest precision. The token-set heuristic should be replaced
   with proper sentence-encoder cosine in a v2 — this is the highest-value
   single improvement to the negative set.

Output: `data/hard_negatives.parquet` (3.8 MB).

## Splits and pair assembly (1B continued)

`src/benchmark/build_splits.py` joins positives with their anchor event,
assigns each pair to three split regimes, and emits pair rows.

Split policy:

| Split | Train | Test | Notes |
|---|---|---|---|
| `split_time` | anchor event date < 2025-01-01 | ≥ 2025-01-01 | excl = no datable event |
| `split_event` | MERGERS_AND_ACQUISITIONS / CHANGE_LEGAL_NAME / CHANGE_LEGAL_FORM | SPINOFF / DEMERGER / BREAKUP / ABSORPTION | excl = no qualifying event |
| `split_jurisdiction` | predecessor country in {US, GB, DE} | {SG, JP, BR} | excl = other countries |

Each positive gets up to 5 hard negatives if its splits are all `train`,
or up to 10 if any split places it in `test`. Anchor is the predecessor;
candidate is the successor (label=1) or a mined negative (label=0).

Resulting `pairs.parquet`:

| Quantity | Value |
|---|---:|
| Positives (label=1) | 31,824 |
| Negatives (label=0) | 93,695 |
| Total pair rows | 125,519 |
| Size on disk | 2.8 MB |

Split coverage (positives + negatives, per split assignment):

| | train | test | excl |
|---|---:|---:|---:|
| `split_time` | 89,676 | 18,080 | 17,763 |
| `split_event` | 93,408 | 14,471 | 17,640 |
| `split_jurisdiction` | 30,096 | 726 | 94,697 |

Note: the jurisdiction test split is small (~726 rows). GLEIF's
single-jurisdiction coverage is sparse in SG / JP / BR vs US / GB / DE; this
mirrors a real-world generalization challenge but reviewers will reasonably
ask for either (a) more jurisdictions in test or (b) a different
held-out-jurisdiction definition. Document this in the paper limitations.

## Wikidata marquee supplement (1C)

`src/benchmark/wikidata_marquee.py` queries Wikidata SPARQL for company
successor pairs (P156) restricted to instances of business / organization,
then matches both sides to GLEIF entities by normalized name.

| Quantity | Value |
|---|---:|
| SPARQL bindings | 4,240 |
| Unique labeled pred → succ pairs | 3,590 |
| Both sides matched to GLEIF | 302 |
| Predecessor only matched | 230 |
| Successor only matched | 838 |
| Neither matched | 2,220 |

After dropping 37 self-loops (Wikidata's predecessor and successor resolve
to the same GLEIF LEI — typically administrative renames), **265 useful
marquee pairs** remain. Of those:

- **19 already in `successor_links.parquet`** — sanity check that the
  multi-source labeling is consistent.
- **246 are net-new** signal that GLEIF's `SuccessorEntity` field doesn't
  capture.

Notable matched pairs include Sprint Corporation → T-Mobile US, Chemical
Financial Corporation → TCF Financial, Cannon Group → MGM, Interbrew →
Anheuser-Busch InBev, Meredith → Gray Media, AT&T Corporation → AT&T,
Salzgitter Mannesmann → Mannesmann.

This is excellent evidence for the multi-source approach: GLEIF gives broad
machine-readable coverage; Wikidata adds hand-curated depth on famous cases.
Output: `data/wikidata_marquee.parquet` (0.2 MB) — kept as a separate
marquee test slice; can be merged into `pairs.parquet` as positives or held
as a qualitative eval set.

## TemporalLink v1 — final artifact summary

The complete v1 dataset deliverables sitting under `data/` (gitignored):

| File | Rows | Size |
|---|---:|---:|
| `entities.parquet` | 3,313,039 | 275.5 MB |
| `events.parquet` | 1,170,716 | 22.7 MB |
| `relationships.parquet` | 473,839 | 10.1 MB |
| `successor_links_raw.parquet` | 33,645 | 1.1 MB |
| `successor_links.parquet` (enriched) | 33,652 | 2.6 MB |
| `hard_negatives.parquet` | 253,733 | 3.8 MB |
| `pairs.parquet` (the v1 dataset) | 125,519 | 2.8 MB |
| `wikidata_marquee.parquet` (supplement) | 3,590 | 0.2 MB |

End-to-end reproducible from the GLEIF XML + a SPARQL endpoint, no
proprietary access required. Total wall time from raw XML to v1 dataset:
~7 minutes.
