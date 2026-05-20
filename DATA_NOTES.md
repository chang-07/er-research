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
