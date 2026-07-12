# Paper artifacts

This directory contains the draft paper, figures, datasheet, and bibliography
for "Sylvester: A Benchmark and Method for Lifecycle-Aware Entity
Resolution on Open Financial Data."

## Files

| File | What |
|---|---|
| `draft.tex` | Full LaTeX source. Uses `article` class as a placeholder; swap to `neurips_2026.sty` (or the venue's style file) at submission time. |
| `refs.bib` | Bibliography. |
| `datasheet.md` | Datasheet for Datasets (Gebru et al.) — required appendix for NeurIPS D&B and ACL Data tracks. |
| `figures/` | Vector PDFs (LaTeX inputs) and PNG previews (for quick viewing on GitHub). |

## Compiling

No LaTeX engine is bundled. Three painless options:

1. **Overleaf**: drop `draft.tex` + `refs.bib` + `figures/` into a fresh
   project; it just works.
2. **MacTeX / TeXLive**: `brew install --cask mactex` then
   `pdflatex draft && bibtex draft && pdflatex draft && pdflatex draft`.
3. **Docker**: `docker run --rm -v "$PWD":/workdir thomasweise/texlive
   pdflatex draft.tex`.

## Regenerating figures

```bash
python3 src/eval/extended.py --data data/        # writes the CSVs
python3 src/eval/figures.py   --data data/       # writes paper/figures/*.pdf
```

The figure CSV inputs (`event_year_histogram.csv`, `eval_metrics.csv`,
`per_event_type_metrics.csv`) are produced by the eval scripts and are
gitignored — regenerate them from `data/pairs.parquet` + the prediction
files in `data/predictions/`.

## Submission-time checklist

- [ ] Swap `\documentclass{article}` for the venue style file.
- [ ] Update author block (currently single-author placeholder).
- [ ] Fill in any anonymous-review redactions if double-blind.
- [ ] Re-render figures at the venue's required DPI.
- [ ] Add the datasheet as an appendix or supplementary file.
- [ ] Make sure all `\cite{}` keys resolve.
- [ ] Compile twice for cross-references.
- [ ] Add author affiliations + acknowledgments.
- [ ] If targeting NeurIPS D&B: ensure benchmark is on Hugging Face
      Datasets before camera-ready.
