# Datasheet for Sylvester v1

Following the template of Gebru et al. (2018), *Datasheets for Datasets*.
This datasheet accompanies the paper "Sylvester: A Benchmark and Method
for Lifecycle-Aware Entity Resolution on Open Financial Data."

---

## 1. Motivation

**For what purpose was the dataset created?**
To enable training and evaluation of entity-resolution systems that must
match legal-entity records across lifecycle events (mergers, acquisitions,
spinoffs, name changes, jurisdiction migrations, etc.). Existing ER
benchmarks evaluate point-in-time matching only; this dataset specifically
exposes the lifecycle-bridge problem, which causes silent failures in
fintech and quant-finance production pipelines.

**Who created the dataset?**
Sheng Chang Li (sli3244@uwo.ca), single author, derived entirely from public
data sources (no proprietary data). The Wikidata supplement uses public
SPARQL queries; the GLEIF labels are CC0; SEC EDGAR data is in the public
domain.

**Who funded the creation of the dataset?**
Independent research; no external funding.

---

## 2. Composition

**What do the instances represent?**
Pairs of legal-entity records, identified by Legal Entity Identifier (LEI),
labeled either as a `successor` relationship (label=1, "these two records
describe the same continuing economic entity across a lifecycle event") or
as `non-successor` (label=0, "these two records describe distinct economic
entities").

**How many instances are there in total?**

| Artifact | Rows |
|---|---:|
| `entities.parquet` (unique LEIs) | 3,313,039 |
| `events.parquet` (per-LEI lifecycle events) | 1,170,716 |
| `relationships.parquet` (Level-2 RR edges) | 473,839 |
| `successor_links.parquet` (enriched + classified) | 33,652 |
| `hard_negatives.parquet` (v1 mined hard negatives) | 253,733 |
| `hard_negatives_v2.parquet` (v2: E5-cosine; recommended for training) | 652,764 |
| `hard_negatives_v3.parquet` (v3: TF-IDF char-3gram; for the paper's miner-robustness evaluation) | 652,680 |
| **`pairs.parquet` (the headline evaluation pool — the paper's benchmark pool; token-overlap-mined negatives)** | **125,519** |
| `pairs_v3.parquet` (the paper's "TF-IDF pool", Sylvester-TF-IDF — alternative evaluation pool with TF-IDF negatives) | 217,957 |
| &nbsp;&nbsp;&nbsp;&nbsp;of which positives (label=1) | 31,824 |
| &nbsp;&nbsp;&nbsp;&nbsp;of which negatives (label=0) | 93,695 |
| `wikidata_marquee.parquet` (qualitative supplement) | 3,590 |

**Does the dataset contain all possible instances or is it a sample?**
The benchmark contains the entirety of the lifecycle-spanning positives in
GLEIF's `SuccessorEntity` field that have a resolvable successor LEI in
the same snapshot (33,645 pairs before filtering). Hard negatives are
mined from the full universe of 3.3M GLEIF entities and up to 4 candidate
classes per positive. The Wikidata supplement is the result of a single
SPARQL query and is bounded by the 8K-result limit on the Wikidata
endpoint.

**What data does each instance consist of?**
Each `pairs.parquet` row has: `anchor_lei`, `candidate_lei`, `label`,
`source`, `anchor_event_date`, `anchor_event_type`, `anchor_country`, and
three split assignments (`split_time`, `split_event`, `split_jurisdiction`,
each `train`/`test`/`excl`). Anchor and candidate refer back to
`entities.parquet`, which contains legal name, jurisdiction, legal form,
registration metadata, addresses, etc.

**Is there a label or target?**
Yes; binary `label` ∈ {0, 1}. Positives are derived from GLEIF's
`SuccessorEntity` field; negatives are mined hard candidates.

**Is any information missing?**
- 16,557 `SuccessorEntity` records in the source snapshot reference a
  successor entity that never registered an LEI; these are excluded from
  the v1 evaluation (but counted in `DATA_NOTES.md`).
- Some entities have non-null `legal_form_code` or address fields.
  Approximately 0.3% of rows have missing `legal_name`; these are
  excluded during pair construction.
- The `split_jurisdiction` test set (88 anchors / 161 positives / 726
  total rows) is small relative to the other splits.

**Are relationships between individual instances made explicit?**
Yes; `relationships.parquet` (Level-2 RR) and `successor_links.parquet`
both encode inter-instance links.

**Are there recommended splits?**
Yes; three split regimes are provided (`split_time`, `split_event`,
`split_jurisdiction`). See Section 3.4 of the paper for definitions.

**Are there any errors, sources of noise, or redundancies?**
- Label noise: ~5.4% of raw `SuccessorEntity` pairs are administrative
  re-issuances rather than real lifecycle events; we filter these out
  using the `likely_administrative` heuristic (`DATA_NOTES.md`,
  Phase-0 spike).
- Date hygiene: 17,763 pair rows (14.2%) have a null
  `anchor_event_date` (the anchor event carries no effective date);
  these are assigned `excl` in `split_time` by design and participate
  only in the event/jurisdiction regimes. 14 rows carry implausible
  pre-1900 effective dates from upstream GLEIF typos (e.g., year
  0201); they fall in the time-split train partition and are harmless
  at this count. Consumers parsing dates should use a coercing parser.
- Coverage skew: GLEIF data quality jumped substantially in 2022
  ("2022 cliff" in `DATA_NOTES.md`). Pre-2022 events are sparse and
  biased.
- Hard-negative noise: the `token_overlap_same_country` fuzzy-fallback
  class (92% of v1 hard negatives) is the lowest-precision; the ablation
  in the paper shows it slightly hurts training.

**Is the dataset self-contained?**
No; the entity records derive from the GLEIF Level-1 LEI-CDF file
distributed by GLEIF under CC0. We provide all parsing scripts and a
fixed snapshot date so the dataset is exactly reproducible by anyone with
access to the GLEIF golden-copy archive (free, no auth).

**Does the dataset contain confidential data?**
No. All sources are publicly published legal-entity records.

**Could the dataset contain offensive, insulting, etc. content?**
Unlikely; legal-entity names are formal corporate identifiers.

---

## 3. Collection Process

**How was the data acquired?**
1. GLEIF Level-1 + Level-2 concatenated XML files downloaded from the
   public golden-copy CDN.
2. SEC EDGAR `company_tickers.json` fetched via HTTP (compliant
   User-Agent).
3. Wikidata pairs fetched via a single SPARQL query on
   `https://query.wikidata.org/sparql`.

**What collection mechanisms or procedures were used?**
Streaming XML parsing with `lxml.iterparse` for GLEIF; JSON download for
SEC; SPARQL HTTP POST for Wikidata. All processing is reproducible from
the included Python scripts in `src/ingest/` and `src/benchmark/`.

**If the dataset is a sample, what was the sampling strategy?**
The 100-row label-quality validation sample is uniform random with
`seed=42` over the 33,645 raw successor pairs. Hard negatives are mined
deterministically up to a configurable cap per predecessor.

**Who was involved and how were they compensated?**
Single author; no external labelers.

**Over what timeframe was the data collected?**
Source snapshot is 2026-05-19. The lifecycle events in the data cover
2014–2026 (with the bulk of high-quality events post-2022).

**Were ethical review processes conducted?**
Not applicable (no human subjects; public legal-entity data only).

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing of the data done?**
Yes. Lifecycle pairs were classified into five categories using rules over
the predecessor's documented event types, name delta after normalization
(lowercase, strip punctuation, drop legal-suffix tokens), jurisdiction
match, and legal-form match. Pairs classified as
`likely_administrative` were filtered out of the v1 benchmark.

**Was the raw data saved in addition to the preprocessed data?**
Yes; `successor_links_raw.parquet` retains the unprocessed pairs.

**Is the software used to preprocess/clean the data available?**
Yes; all preprocessing is the Python code in `src/`.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes; the accompanying paper trains and evaluates the similarity, LLM,
bi-encoder, and cross-encoder baselines and methods reported in its
Table 1 on this dataset.

**Is there a repository linking to other papers/systems using the
dataset?**
The code release at https://github.com/chang-07/er-research is the
canonical entry point.

**What other tasks could the dataset be used for?**
- Training cross-encoder or LLM ER systems on lifecycle bridges.
- Studying concept drift in entity matching (the 2014–2026 time span
  enables explicit temporal generalization studies).
- Studying jurisdiction generalization in financial ML (held-out-
  jurisdiction split).
- Benchmarking corporate-graph completion methods (using the Level-2 RR
  edges as side information).

**Is there anything about the composition or collection that may impact
future uses?**
- The 2022 cliff in event-volume means models trained on pre-2022 data
  may underperform on post-2022 evaluation (and vice-versa).
- Coverage in emerging-market jurisdictions is much thinner than US/EU.
  Conclusions about jurisdiction generalization should be qualified.

**Are there tasks for which the dataset should not be used?**
- Predicting future M&A events (the dataset captures completed events
  with effective dates, not preannouncement signals).
- Inferring private-company financial information beyond what is in the
  GLEIF / EDGAR / Wikidata public record.

---

## 6. Distribution

**Will the dataset be distributed to third parties?**
Yes; the code release is public on GitHub. The intermediate parquet
artifacts are not committed (gitignored by `.gitignore`) but are exactly
reproducible from the included scripts in ~7 minutes from the raw GLEIF
XML.

**How will the dataset be distributed?**
- Code: GitHub (`chang-07/er-research`, MIT or Apache 2.0).
- Derived data: planned upload to Hugging Face Datasets after paper
  submission.

**When will the dataset be distributed?**
At paper submission / arXiv preprint posting.

**Will the dataset be distributed under a copyright or other IP
license?**
Code: MIT or Apache 2.0 (TBD). Derived data: CC-BY-4.0 (GLEIF source is
CC0; Wikidata is CC0; SEC is public domain; the curation, splits, and
hard-negative selection are the contributed work).

**Have any third parties imposed IP-based or other restrictions?**
No.

**Do any export controls or other regulations apply?**
No.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
Sheng Chang Li (sli3244@uwo.ca). Issues should be filed at the GitHub
repository.

**How can the owner/curator be contacted?**
sli3244@uwo.ca or via GitHub.

**Is there an erratum?**
No errata at v1 release. Updates will be tracked in the GitHub
repository's `CHANGELOG.md` and on Hugging Face.

**Will the dataset be updated?**
v2 and v3 additions have already shipped alongside v1:
- **v2** (`data/hard_negatives_v2.parquet`): re-mines the dominant
  same-country hard-negative class using sentence-encoder (E5-small)
  cosine, reducing label noise from the v1 token-overlap fallback.
  Cross-encoder training uses v2 negatives.
- **v3** (`data/hard_negatives_v3.parquet`, `data/pairs_v3.parquet`):
  alternative hard-negative pool from char-3gram TF-IDF cosine ---
  structurally orthogonal to v2 (no learned representation, no shared
  parameters with E5). This is the paper's "TF-IDF pool"
  (Sylvester-TF-IDF), used for its miner-robustness evaluation;
  consumers can use either pool depending on which similarity-signal
  they want the benchmark to test against.

Future planned additions (not yet shipped):
- EDGAR 8-K Item 5.03 supplementary positives for US public companies.
- Expanded held-out-jurisdiction split (current SG/JP/BR pool is thin
  at n=88 anchors; planned to add KR, MX, AR, CL, ID).
- Historical GLEIF snapshots to materialize identity positives (same
  entity at different times, no lifecycle event between).

**If the dataset relates to people, are there applicable limits on the
retention of the data associated with the instances?**
The dataset relates only to legal entities (companies, funds, branches),
not natural persons. GLEIF records do not contain personal data.

**Will older versions continue to be supported/hosted/maintained?**
Yes; the v1 snapshot is permanently pinned via the included
`20260519-…` filename convention and the reproducibility scripts.

**If others want to extend/augment/build on the dataset, is there a
mechanism for them to do so?**
Yes; pull requests welcome at the GitHub repository.
