# Sylvester

**Lifecycle-aware entity resolution for financial entities.**

Most entity-resolution (ER) systems answer "do these two records refer to the same
entity *right now*?" They fall apart when the entity has undergone a lifecycle
event — a merger, spinoff, rebrand, ticker change, or jurisdiction migration —
between the two records. This repo builds an open benchmark for that problem (to
our knowledge the first) and a method that beats both classical record-linkage
and modern LLM pair-matching baselines on lifecycle-spanning pairs.

Everything is built from open, CC0/CC-BY registries (GLEIF, SEC EDGAR, Wikidata).
No proprietary sources required; the full pipeline runs end-to-end on a laptop.

---

## TL;DR — the findings in five lines

1. **The task is real and hard.** Hard negatives are mined to be *name-similar* to
   each predecessor; true successors are by construction *name-different*. So on
   this benchmark, name similarity is **evidence against** a match — every
   similarity baseline lands **below 0.5 AUROC** on the time/event splits.
2. **Among name-input methods, joint encoding is the only survivor.** A 22M-param
   MiniLM **L-6 cross-encoder** reaches **0.607 / 0.569 / 0.750 Hits@1** across
   TIME / EVENT / JURIS — **+35 / +27 / +24 pp** over the best bi-encoder, above
   all six zero-shot LLMs tested (4 orgs), and it alone transfers across all
   three pools with name+state inputs. An L-6×L-12 ensemble reaches
   **0.700 / 0.677 / 0.784**.
3. **The benchmark "cuts both ways" — and the ceiling is the input, not the
   architecture.** Anti-similarity (negated Jaccard) scores **0.700 / 0.638 /
   0.516** on-pool and a name-feature GBM tops every name-only method
   (0.772 / 0.734 / 0.835); both collapse off-pool. But a GBM trained on the
   cross-encoder's own E5 negatives **with the full structured record**
   (liveness, addresses, previous names, dates) **transfers everywhere and
   wins**: 0.783 / 0.753 / 0.881 on-pool (+18 pp over the L-6, p<1e-4),
   0.78 / 0.75 cross-miner, 0.51–0.66 fielded retrieval-conditional (n.s. vs
   the enriched ensemble). The lifecycle signal lives in the record, not the
   name.
4. **The honest numbers come from off-pool tests.** Both exploits **collapse**
   off-pool (GBM: 0.27–0.36 on a different-miner pool, 0.12–0.15 fielded);
   within the name-input class the cross-encoder stays strong
   **simultaneously** on every pool, and the full-record GBM transfers with the
   best minimum of all. On a **fielded retrieve-then-rerank** pipeline over
   the full 3.3M-entity universe, end-to-end Hits@1 is **15–33%** (recall-bottlenecked) —
   that is the deployment number, not the curated one.
5. **Known caveats are quantified, not hidden** (see [Honest caveats](#honest-caveats)):
   the EVENT split is 94.5% absorption, JURIS has only n=88 (directional), GLEIF
   labels carry survivorship bias, and all three negative miners share a name axis.

> **Naming map (repo ↔ paper).** Internal labels predate the paper's final
> terminology. The *v2 cross-encoder* is the paper's headline **L-6 cross-encoder
> (22M)**; *v8aL12* is the **L-12 (33M)** variant; *v10* is the **L-6 × L-12
> ensemble**. *v1 / v2 hard negatives* are the **token-overlap** and **E5-cosine**
> miners; `pairs.parquet` is the paper's **benchmark pool** (token-overlap-mined
> negatives) and `pairs_v3.parquet` its **TF-IDF pool** (Sylvester-TF-IDF). Artifact, script, and model-directory
> names keep the internal labels so the reproduction commands stay valid.

---

## Repository layout

```
gleif_audit.py                  Phase-0 viability audit on the raw GLEIF dump
validate_successor_pairs.py     Phase-0 label-quality spike
lei_cik_crosswalk.py            Phase-0 LEI↔CIK (SEC EDGAR) feasibility spike
src/
  ingest/                       XML → typed parquet (Level-1 entities/events, Level-2 RR)
  benchmark/                    label building, hard-neg mining (v1/v2/v3), splits,
                                transitive positives, Wikidata, fielded retriever
  baselines/                    cheap (exact/token/tfidf), E5 encoder, LLM pairwise
  method/                       bi-encoder + cross-encoder training, scoring, ablations
  eval/                         evaluate.py + diagnostics (gap_checks, calibration,
                                error taxonomy, cost/latency, robustness CIs, …)
paper/                          draft.tex, refs.bib, datasheet.md (Gebru), figures/
data/                           all parquet artifacts + predictions/ + models/ (gitignored)
PLAN.md                         the full 12-week plan and task ledger
DATA_NOTES.md                   chronological lab notebook: every count, every phase
```

---

## Procedures

The benchmark is built in five stages; the method and the robustness/honesty
checks follow. Each stage is one or two scripts and reproduces from raw XML in
~7 minutes of wall time.

### 1. Data sources

| Source | What we take | Scale |
|---|---|---|
| **GLEIF Level-1** (LEI-CDF concatenated file, CC0) | legal name, jurisdiction, legal form, addresses, `LegalEntityEvents`, and the `SuccessorEntity` predecessor→successor link | 3.31M entities, 1.17M events |
| **GLEIF Level-2** (Relationship Records) | typed parent/subsidiary, fund-management, branch edges — used for hard negatives | 473,839 relationships |
| **SEC EDGAR** (`company_tickers.json`) | name↔CIK crosswalk for the US public-company slice | 3,334 confident links |
| **Wikidata** (SPARQL P156 *followed by*) | marquee, hand-curated M&A as a supplement / cross-check | 302 both-side-matched pairs |

Snapshot: **2026-05-19**. Ingestion is a single streaming `lxml.iterparse` pass
(`src/ingest/`) producing `entities.parquet`, `events.parquet`,
`relationships.parquet` — ~302 MB, a 27× compression of the 7.5 GB source XML.

### 2. Positive labels

Gold positives are GLEIF `SuccessorEntity` links (predecessor-LEI → successor-LEI),
maintained by GLEIF's Local Operating Units — direct supervision, no annotation.
Of 50,202 records with a successor element, **33,645** have a resolvable successor
LEI. A five-class auto-classifier (documented event type + normalized name delta +
jurisdiction + legal form) keeps the four "plausibly-real" classes (**94.6%**) and
drops `likely_administrative` (5.4%), yielding **31,817 predecessors / 31,824
positive pairs**. (`src/benchmark/build_successor_links.py`)

### 3. Hard negatives

For each positive predecessor, up to 80 hard negatives across four classes
(`src/benchmark/mine_hard_negatives.py`, deduped to **253,733** pairs):

| Class | Definition | Count |
|---|---|---:|
| `token_overlap_same_country` | ≥2 shared content tokens, same country (fuzzy fallback) | 232,601 |
| `same_name_diff_country` | identical normalized name, different country | 10,703 |
| `same_name_same_country` | identical normalized name, same country | 10,208 |
| `parent_subsidiary` | shares a Level-2 RR edge — looks related, legally distinct | 221 |

Known successors are excluded from the negative set to avoid label contamination.
**v2** (`mine_hard_negatives_v2.py`) replaces the noisy fuzzy class with
**E5-cosine** nearest-neighbor mining (652K negs, full predecessor coverage);
**v3** (`mine_hard_negatives_v3.py`) mines a structurally orthogonal pool with
**char-3gram TF-IDF** cosine — used for the miner-robustness experiment below.

### 4. Evaluation splits

Each positive is assigned by its *anchor lifecycle event* to three held-out
regimes (`src/benchmark/build_splits.py`); negatives follow their predecessor.
Up to 5 negs/positive if all splits are train, up to 10 if any split is test →
**125,519 pair rows** (`pairs.parquet`).

| Split | Train | Test | Test size |
|---|---|---|---|
| **TIME** | anchor date < 2025-01-01 | ≥ 2025-01-01 | ~2,007 eligible anchors |
| **EVENT** | M&A / rename / legal-form | spinoff / demerger / breakup / **absorption** | ~1,612 anchors (94.5% absorption) |
| **JURIS** | US + GB + DE | SG + JP + BR | 88 anchors (directional, ±8.8 pp CI) |

### 5. Baselines → method

- **Cheap** (`src/baselines/cheap.py`): exact-name, token-set Jaccard, char-3gram TF-IDF.
- **Encoder** (`encoder.py`): zero-shot `intfloat/e5-small-v2` cosine.
- **LLM pairwise** (`llm_pairwise.py`): single-pair 0–10 rubric over Anthropic /
  OpenAI / Gemini / local Ollama. *Drives paid APIs — run manually with a key.*
- **Bi-encoder** (`src/method/train.py`): E5-small fine-tuned with state features +
  hard negatives (InfoNCE). Modest +5–8 pp over zero-shot.
- **Cross-encoder** (`src/method/train_cross_encoder.py`): `ms-marco-MiniLM-L-6-v2`
  (22M, joint encoder) — **the headline method**. Joint attention over the *pair*
  escapes the fixed-embedding ceiling that caps the bi-encoder on this anti-correlated task.
- **Ensemble**: average of the L-6 and an L-12 (33M) cross-encoder (`v4_progression.py`).

### 6. Robustness & honesty checks

- **Anti-similarity baseline** (`gap_checks.py`): rank by *negated* similarity to
  measure how much the construction can be exploited directly.
- **2×2 miner-robustness matrix** (`compare_distributions.py` + `compute_robustness_cis.py`):
  train on E5-neg vs TF-IDF negs × evaluate on both pools — isolates how much of the
  gain is a within-miner artifact (answer: 2–13 pp; the rest survives, +13 to +39 pp cross-pool).
- **Fielded retrieve-then-rerank** (`fielded_retriever.py`): bi-encoder retrieves
  top-20 from the full 3.3M universe (same-country filter), cross-encoder reranks —
  the deployment-realistic number.
- **Gap checks** (`gap_checks.py`, `gap_checks2.py`): chance level, train/test entity
  overlap + a memorization check, cross-jurisdiction recall ceiling, date hygiene,
  per-event-type exploit transfer, temperature scaling.
- Plus calibration (ECE/Brier), B-cubed cluster metrics, a named error taxonomy, a
  secondary orthogonal label-quality check, and cross-dataset transfer (Magellan).

---

## Findings

### Main results — test-set Hits@1 (and AUROC)

| Method | TIME | EVENT | JURIS | TIME AUROC | EVENT AUROC | JURIS AUROC |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| zero-shot E5-small | 0.204 | 0.232 | 0.432 | 0.315 | 0.386 | 0.648 |
| v2 bi-encoder | 0.250 | 0.296 | 0.500 | 0.405 | 0.457 | 0.694 |
| **L-6 cross-encoder** (headline name-input) | **0.607** | **0.569** | **0.750** | **0.836** | **0.825** | **0.868** |
| L-6×L-12 ensemble | 0.700 | 0.677 | 0.784 | — | — | — |
| *anti-similarity exploit* (on-pool only) | *0.700* | *0.638* | *0.516* | *0.807* | *0.734* | *0.677* |
| **features_gbm-full** (full record, transfers) | **0.783** | **0.753** | **0.881** | 0.929 | 0.912 | 0.893 |
| **Fielded end-to-end** (L-6 / GBM-full) | **0.15–0.28 / 0.17–0.33** | | | | | |

Three things to read off this table:

1. **Every similarity baseline is sub-0.5 AUROC on TIME/EVENT** — *worse than random* —
   because the construction makes name similarity anti-predictive. This is the result
   that justifies the benchmark, and it is a property of the *construction*, not a
   universal claim about ER.
2. **Among name-input methods, the cross-encoder is the only one that transfers.**
   Joint encoding asks whether the *differences* between two records look like a
   lifecycle-event pattern. Supervised GBMs post higher on-pool AUROC, but the
   name-only ones collapse off-pool — only the full-record variant transfers.
3. **Curated Hits@1 is gameable** — the anti-similarity exploit beats the L-6 on
   TIME/EVENT on-pool. Within the name-input class, the cross-encoder's claim is
   being strong *simultaneously* on all three pools; the full-record GBM shows the
   remaining headroom is input, not architecture.

### The exploit collapses off-pool (verified via `gap_checks2.py`)

| Pool / slice | anti-similarity Hits@1 | cross-encoder Hits@1 |
|---|:-:|:-:|
| Curated E5-neg pool (TIME/EVENT/JURIS) | 0.677 / 0.607 / 0.500 | 0.605 / 0.564 / 0.750 |
| TF-IDF pool (different miner) | 0.191 / 0.193 / 0.155 | 0.545 / 0.528 / 0.571 |
| Fielded pool (conditional) | 0.193 / 0.171 / 0.087 | 0.538 / 0.520 / 0.478 |
| Pure renames (M&A 0.72) | **0.10** | 0.90 |

The exploit is a one-way bet on acquirer-renaming; the cross-encoder holds across
all of them. (These reproduce the paper's "construction cuts both ways" analysis,
§ _The construction cuts both ways: an anti-similarity baseline_.)

### Honest caveats

- **In-distribution headline overstates deployment.** Curated 56–61% (TIME/EVENT)
  drops to **15–33% end-to-end** on a fielded pipeline — the bottleneck is
  first-stage **bi-encoder Recall@20 (33–50%)**, not the reranker.
- **"Held-out event type" ≈ "held-out absorption."** The EVENT test set is 94.5%
  absorption; demerger / breakup / spinoff are present but thin (n=42 / 20 / 4). The
  +34 pp aggregate is primarily an absorption-transfer claim.
- **JURIS is small (n=88) and directional** (±8.8 pp bootstrap CI). Read absolute
  per-split numbers with sample sizes in mind.
- **Survivorship / reporting bias.** GLEIF `SuccessorEntity` is voluntarily populated
  (biases toward well-counseled regulated entities); 16,557 successors that never
  registered an LEI are excluded.
- **Splits are pair-level, not entity-disjoint** (32% / 21% of TIME-split test
  candidates / golds appear in train pairs). But a memorization check shows L-6
  Hits@1 on *unseen* golds is **no lower** (0.618 vs 0.566 on TIME) — leakage is not
  driving the result. Entity-disjoint splits are a stricter future design.
- **All three miners share a name axis** (token-overlap, E5-cosine, TF-IDF char-3gram).
  The one non-name negative class (`parent_subsidiary`, n=221) is both tiny and the
  only class the cross-encoder regresses on (0.45→0.78 mean score) — the fielded
  result is the genuinely name-orthogonal evidence.

---

## Reproducing

Requires the latest GLEIF Level-1 and Level-2 golden-copy XML (from
<https://goldencopy.gleif.org/>, published every 8 hours, CC0). Place them at the
project root, then `pip install -r requirements.txt`.

### Phase 0 — viability spikes
```bash
python3 gleif_audit.py              # ~50s — viability audit
python3 validate_successor_pairs.py # ~90s — label-quality check
python3 lei_cik_crosswalk.py        # ~50s — LEI↔CIK feasibility (edit UA email first)
```

### Phase 1 — build Sylvester v1 (raw XML → benchmark, ~7 min)
```bash
python3 src/ingest/gleif_to_parquet.py    --xml <Level1.xml> --out data/   # ~160s
python3 src/ingest/gleif_rr_to_parquet.py --xml <RR.xml>     --out data/   # ~10s
python3 src/benchmark/build_successor_links.py --data data/                # ~3s
python3 src/benchmark/mine_hard_negatives.py   --data data/                # ~5 min
python3 src/benchmark/build_splits.py          --data data/                # ~10s
python3 src/benchmark/wikidata_marquee.py      --data data/                # ~30s
```

### Phase 2 — baselines
```bash
python3 src/baselines/cheap.py   --data data/   # exact_name, token_set, tfidf_char
python3 src/baselines/encoder.py --data data/   # E5-small-v2 zero-shot
python3 src/baselines/features_gbm.py --data data/  # supervised GBM over string+state features
                                                    # (trains no-leakage; scores all 3 pools)
python3 src/benchmark/build_pairs_v2.py --data data/     # E5-negative pool (matches CE training negs)
python3 src/baselines/features_gbm.py --data data/ --train-pairs pairs_v2.parquet \
    --features full --out-suffix _trnv2full         # full-record GBM (the de-confounded variant)
python3 src/eval/eval_gbm_sweep.py data/            # score the sweep on all 3 pools + fielded views
python3 src/eval/evaluate.py     --data data/   # AUROC, AUPRC, Hits@K, MRR
```
LLM pairwise drives paid APIs and is **not** run autonomously. It needs a key
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`). The runner is
money-safe by default: it prints a cost estimate and asks for confirmation
(`--yes` to skip), checkpoints every result to a JSONL log as it lands, resumes
interrupted runs with `--resume` (only unscored pairs are re-billed), pins
`temperature=0`, retries transient errors with backoff, and fail-fasts after 15
consecutive hard failures (bad key / no credits). Failed pairs are excluded from
the output — never silently scored 0 — and the eval scripts drop anchors with
incomplete candidate pools.
```bash
# scope + cost preview (no key needed)
python3 src/baselines/llm_pairwise.py --dry-run --test-anchors-per-split 2100
# tiny paid smoke test (~$0.02), then full eval coverage (~26.6K calls)
python3 src/baselines/llm_pairwise.py --test-anchors-per-split 2 --out-name llm_haiku_smoke
python3 src/baselines/llm_pairwise.py --test-anchors-per-split 2100 --out-name llm_haiku_full
python3 src/eval/eval_llm_sample.py --pred llm_haiku_full          # per-model Hits@1/MRR
python3 src/eval/eval_llm_panel.py  --preds llm_haiku_full ...     # cross-model, shared anchors
```

### Phase 3 — bi-encoder + ablations
```bash
python3 src/method/train.py --data data --variant full       # name + jurisdiction + form
python3 src/method/train.py --data data --variant no_state   # ablation
python3 src/method/train.py --data data --variant no_hard    # ablation
python3 src/eval/evaluate.py --data data
```
3 epochs, batch 32, lr 2e-5, Apple MPS (~6 min train + ~50s inference per variant).
Default training set is the ~26K positives that are `train` in **all three** splits
(strict no-leakage).

### v2 — E5-cosine negs + transitive positives + cross-encoder (~25 min)
```bash
python3 src/benchmark/mine_hard_negatives_v2.py --data data/   # E5-cosine negs (~7 min)
python3 src/benchmark/transitive_positives.py   --data data/   # A→B→C chains (~3s)
python3 src/method/train.py --variant v2_full   --data data/   # v2 bi-encoder
python3 src/method/train_cross_encoder.py       --data data/   # headline L-6 cross-encoder
python3 src/eval/evaluate.py --data data/
```

### v3 — miner-robustness 2×2
```bash
python3 src/benchmark/mine_hard_negatives_v3.py --data data/                          # TF-IDF negs
python3 src/benchmark/build_splits.py --data data/ \
    --hard-negs hard_negatives_v3.parquet --out-name pairs_v3.parquet
python3 src/method/train_cross_encoder.py --data data/ \
    --hard-negs hard_negatives_v2.parquet --train-pairs pairs.parquet \
    --extra-pairs pairs_v3.parquet --out-name method_v2_crossenc_repro               # train-E5
python3 src/method/train_cross_encoder.py --data data/ \
    --hard-negs hard_negatives_v3.parquet --train-pairs pairs_v3.parquet \
    --extra-pairs pairs.parquet --out-name method_v3_crossenc                        # train-TFIDF
python3 src/eval/compare_distributions.py --data data/ --skip-ci
python3 src/eval/compute_robustness_cis.py
```

### Diagnostics & robustness (Phase v3.5)
```bash
python3 src/eval/gap_checks.py  data    # anti-sim, chance, entity overlap, recall ceiling
python3 src/eval/gap_checks2.py data    # off-pool exploit transfer + memorization check
python3 src/eval/cost_latency.py        # cost & latency per method
python3 src/eval/calibration.py         # reliability diagram + ECE/Brier
python3 src/eval/cluster_metrics.py     # B-cubed precision/recall
python3 src/eval/error_taxonomy.py      # named failure taxonomy
python3 src/eval/v4_progression.py      # v4→v10 enriched-prompt + ensemble progression
```
> Note: these scripts have no internal time limit; on macOS do **not** prefix them
> with `timeout` (it isn't installed and silently no-ops the run). Run them directly.

---

## Paper, datasheet, and data release

- **Paper:** `paper/draft.tex` (+ `refs.bib`) — ACM `sigconf`, targeting **ICAIF 2026**
  (8 pages incl. refs). Compile notes in `paper/README.md`.
- **Datasheet:** `paper/datasheet.md` (Gebru et al. template).
- **Dataset card / metadata:** `data/HF_README.md`, `data/croissant.json` (HF release
  is a manual upload after submission).

All artifacts under `data/` are gitignored and regenerated by the commands above.
See [DATA_NOTES.md](DATA_NOTES.md) for the full chronological record (every count,
every phase, every runtime) and [PLAN.md](PLAN.md) for the task ledger.

## License

TBD — most likely **MIT** for code, **CC-BY-4.0** for any derived dataset we release
(GLEIF data is itself CC0).
</content>
</invoke>
