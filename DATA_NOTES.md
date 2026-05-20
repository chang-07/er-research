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
