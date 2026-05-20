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
- [ ] Benchmark construction (positive pairs, hard negatives, three eval splits)
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
python3 gleif_audit.py             # ~50s — viability audit
python3 validate_successor_pairs.py # ~90s — label-quality check on successor pairs
python3 lei_cik_crosswalk.py        # ~50s + 1 small SEC fetch
```

Note: the SEC fetch in `lei_cik_crosswalk.py` requires an email-bearing
User-Agent. Edit the `UA` constant before running with your own contact.

## License

TBD — most likely MIT for code, CC-BY-4.0 for any derived datasets we release
(GLEIF data is itself CC0).
