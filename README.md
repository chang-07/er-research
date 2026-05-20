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
- [ ] Baselines (Dedupe, sentence encoders, LLM pairwise matching)
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

## License

TBD — most likely MIT for code, CC-BY-4.0 for any derived datasets we release
(GLEIF data is itself CC0).
