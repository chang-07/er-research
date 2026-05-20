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

- [x] Data-availability audit on the GLEIF Level-1 LEI-CDF dump (run `gleif_audit.py`)
- [ ] Manual validation pass on 100 successor links (label-quality check)
- [ ] LEI ↔ CIK crosswalk spike (EDGAR `company_tickers.json` + OpenFIGI)
- [ ] Benchmark construction (positive pairs, hard negatives, three eval splits)
- [ ] Baselines (Dedupe, sentence encoders, LLM pairwise matching)
- [ ] Method (entity-state-at-time bi-encoder w/ event-bridging contrastive loss)
- [ ] Write-up + arXiv preprint
- [ ] Submit (target: ICAIF 2026 workshop primary, NeurIPS 2026 D&B stretch)

See [PLAN.md](PLAN.md) for the full 12-week plan.
See [DATA_NOTES.md](DATA_NOTES.md) for what the GLEIF audit told us.

## Reproducing the audit

You need the latest GLEIF Level-1 concatenated file (XML, ~2 GB compressed,
~7.5 GB decompressed). Download from
<https://www.gleif.org/en/lei-data/gleif-concatenated-file/download-the-concatenated-file>
and place it at the project root.

```bash
pip install -r requirements.txt
python3 gleif_audit.py
```

Takes ~50 seconds on a modern laptop. Output includes total entity count,
SuccessorEntity link count, event-type breakdown, per-year histogram, and a
go/no-go verdict.

## License

TBD — most likely MIT for code, CC-BY-4.0 for any derived datasets we release
(GLEIF data is itself CC0).
