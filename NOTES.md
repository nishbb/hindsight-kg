# NOTES

Running log of plans, decisions, failures and open issues. Newest milestone at the bottom.

## Compute log

| Milestone | Where | GPU hours | Running total GPU hours | Notes |
|---|---|---|---|---|
| M0 | laptop (Apple M-series, 11 CPU, 36 GB RAM) | 0 | 0 | |
| M1 | laptop | 0 | 0 | about 1 min CPU |

No NVIDIA GPU is attached to this development machine. Everything up to M5 is designed to run on CPU; M6 (QLoRA G-Retriever) needs the Brev RTX 6000 class instance.

## Global decisions

1. **T = ClinVar 2020-01 release** (`variant_summary_2020-01.txt.gz`, FTP Last-Modified 2020-01-02 05:05 GMT). `T_DATE = 2020-01-02`. Every time-T input must be dated on or before T_DATE (`src/kggc/data/sources.py`).
2. **Latest = ClinVar `variant_summary.txt.gz` with Last-Modified 2026-11-16** (md5 `7f64e1f49cf025d0f02757a621971c40`, verified against NCBI's `.md5`). Used only for outcome labels.
3. **Background KG: Hetionet v1.0 (2017) instead of PrimeKG for the temporal experiment.** Two reasonable options:
   * (a) PrimeKG, as the brief suggests. PrimeKG's Dataverse release was built from sources released through 2021 (DisGeNET, HPO, GO, etc.). Its disease-gene edges plausibly encode knowledge created by the same reclassifications we want to predict, so it leaks future information into a 2020 snapshot.
   * (b) Hetionet v1.0, built from sources released by 2017, plus MONDO 2019-11-25 for diseases, HPO 2019 for phenotypes and the 2019-12-16 GWAS Catalog release.
   * Choice: **(b) is the primary, conservative setting.** PrimeKG is kept as a separately labeled "leaky background" sensitivity analysis and is used in M6 because STaRK-Prime is built on it.
4. **Disease identifiers: MONDO v2019-11-25.** Hetionet has only 137 Disease Ontology diseases, far too few for ClinVar conditions, so MONDO 2019 supplies the disease nodes and hierarchy. Hetionet diseases are attached to MONDO via MONDO `equivalentTo` DOID cross-references. Only `equivalentTo` xrefs are used for mapping (conservative); xrefs that map to more than one MONDO term are dropped.
5. **HPO at T.** The original HPO Jenkins server is offline and GitHub releases from 2019 carry no annotation assets. Recovered from the Internet Archive:
   * `ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt` captured 2019-09-02 (150,654 lines, 4,073 genes; complete, last line intact).
   * `hp.obo` at git tag v2019-11-08.
   * **Failure:** every pre-2020 capture of `phenotype_annotation.tab` / `phenotype.hpoa` (disease-phenotype) is truncated (the 2019-07-17 capture stops at exactly 1 MiB). Disease-phenotype edges are therefore **not** included in the time-T graph. Hetionet's Disease-presents-Symptom edges (MeSH symptoms, 2017) are kept instead. Open issue: a complete pre-T disease-phenotype file would add signal.
6. **GWAS Catalog: release 2019-12-16** from the EBI FTP archive, additionally filtered on `DATE ADDED TO CATALOG <= T_DATE`.
7. **Gradient-boosted baseline uses scikit-learn `HistGradientBoostingClassifier`.** LightGBM wheels need a system `libomp` that is absent here; sklearn avoids the system dependency and runs in CI.

## M0. Setup (done)

Plan: package skeleton under `src/kggc`, pyproject with ruff/black/pytest config, pre-commit, CI on CPU, MIT license (already in repo), related-work check.

Built:
* `pyproject.toml`, `environment.yml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (CPU torch, `pytest -m "not network"`).
* `src/kggc/data/cache.py`: disk cache for every API response (`data/cache/<source>/..`) with request, UTC timestamp, source version; per-source rate limits; offline mode (`KGGC_OFFLINE=1`).
* `docs/related_work.md`: 25 verified references. **Finding:** nothing we found does H1 to H3 together. Closest: PathoGN (Kamada et al. 2021, GCN evaluated on 2019 to 2020 ClinVar changes), RENOVO and its 2024 follow-up (tabular VUS reclassification), a 2025 Padova thesis (tabular, leakage-controlled ClinVar 2018 to 2025), MAKGED (multi-agent LLM + GCN KG error detection with synthetic noise), and the 2025 hackathon repo `Model_Garbage_Collection` (GNN flags synthetic errors in Monarch). No stop condition triggered.

## M1. Temporal ClinVar labels (done)

Plan: load T, latest and two pre-T snapshots (2018-01, 2019-01) for history features; filter; build labels; stratified counts; disjointness test.

Filters (counts from `results/m1/filter_steps.csv`):

| Step | Records |
|---|---|
| Records at T (one per VariationID, GRCh38 preferred) | 674,670 |
| OriginSimple exactly `germline` | 598,358 |
| ClinicalSignificance exactly P, LP, P/LP or VUS | 308,770 |
| single gene | 302,761 |
| exactly one informative condition | 181,406 |
| condition maps to one MONDO term at T | 160,813 |
| present in latest | 153,842 |
| latest class in {P/LP, VUS, B/LB, conflicting} | 128,330 |

Labels: downgraded 4,171; stable_pathogenic 50,721; upgraded 2,131; stable_vus 71,307. Downgrade rate among P/LP = 7.6%; upgrade rate among VUS = 2.9%. 5,525 VUS-to-benign are recorded but not labeled. Downgrades went to conflicting (2,461), VUS (1,625) and B/LB (85).

Decisions:
* Exact-match significance only; compound strings (e.g. "Pathogenic, risk factor", "low penetrance") are excluded at both T and latest. Conservative.
* Latest classification is the variant-level aggregate germline classification, not per-condition. Since we require a single condition at T this is usually the same condition, but the latest record may have gained conditions. Logged as a limitation.
* 2018-01 has no VariationID column; mapped via AlleleID from the T release.
* Guard: rows with LastEvaluated after T are dropped (0 found).

Early signal worth noting: "no assertion criteria provided" P/LP records downgrade at 10.8% versus 7.6% overall, and missense P/LP records downgrade far more than truncating ones. Tabular baselines will be strong.
