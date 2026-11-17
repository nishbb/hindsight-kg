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

## M2. Time-T knowledge graph (done)

Plan: MONDO 2019 disease backbone, Hetionet 2017 molecular background, HPO 2019 gene-phenotype, ClinVar 2020-01 variant nodes and assertion edges (significance as edge type), GWAS Catalog 2019-12-16 variant-trait edges; save PyG `HeteroData`; leakage tests.

Built (`scripts/m2_build_kg.py`, 22 s CPU; numbers in `results/m2/summary.json`):
* 12 node types, 450,828 nodes: 353,600 variants (291,591 ClinVar, 62,009 GWAS-only), 21,008 genes, 21,838 diseases (MONDO), 14,586 phenotypes (HPO), 1,926 GWAS traits, plus GO terms, pathways, anatomy, compounds, symptoms.
* 32 edge types, about 3.2M edges. Assertion edges at T: `assoc_plp` 90,390; `assoc_vus` 194,872; `assoc_blb` 74,216; `assoc_conflict` 33,239. GWAS variant-trait 89,603 (from 107,652 genome-wide significant rows added and published on or before T, of 166,103 raw rows).
* Variant node features are intrinsic only: consequence parsed from the HGVS name and variant type. Nothing derived from later data.
* Unlike the M1 label table, the graph keeps multi-condition variants (one assertion edge per mapped condition).

Leakage protection (`src/kggc/graph/leakage.py`, `tests/test_m2_graph.py`):
* every node type and edge type records its source key; the test fails if any source is unknown, flagged leaky, or released after T;
* max ClinVar LastEvaluated, max GWAS date-added and max GWAS publication date in the built graph must be on or before T (observed: GWAS date added 2019-12-06);
* the variant table must not contain any outcome column;
* negative tests confirm the checker rejects PrimeKG, latest ClinVar, post-T dates and non-allow-listed feature columns.

Decisions:
* Hetionet side-effect edges (CcSE) and pharmacologic classes are skipped as irrelevant to genotype-phenotype reasoning.
* 136 of 137 Hetionet diseases map to MONDO through `equivalentTo` DOID xrefs.
* GWAS-only variants get an `in_gene` edge only when the catalog's MAPPED_GENE is a single gene (intergenic "A - B" mappings get none).
* Hetionet itself includes GWAS Catalog and DisGeNET content, but only as of 2017, so it is time-safe.

Open issue: `pyg-lib` has no wheel for this laptop's torch build, so `LinkNeighborLoader` (mini-batch neighbor sampling) cannot run here. M3 therefore has two code paths: neighbor-sampled mini-batches when `pyg-lib` or `torch-sparse` is present (the GPU instance), and full-batch training on CPU otherwise.

## M3. GNN edge plausibility (R-GCN done; HGT running)

Plan: link predictor for `(variant, assoc_plp, disease)`; out-of-sample score for every P/LP edge and VUS pair; split conformal; baselines; H1 test of GNN versus tabular.

Design decisions:
* **Cross-fitting.** Variants are split into 5 folds. For fold k the model never sees any assertion edge of fold-k variants (not in messages, not in supervision), then scores them. Every edge gets an out-of-sample score. 10% of training variants are a validation set for early stopping and conformal calibration.
* **No self-leak in message passing.** Assertion edges only point variant to disease, so a variant never receives a message carrying its own class. Other variants' classes reach it through derived gene-disease edges recomputed from the current message edges. During training 30% of P/LP and B/LB edges are held out of the message graph as supervision each epoch.
* **Negatives.** B/LB assertions of training variants (hard) plus random-disease corruptions. VUS edges are not supervised (two options: treat VUS as negatives, or leave them unlabeled; unlabeled is more conservative because some VUS are truly pathogenic).
* **GNN training uses time-T information only.** Later outcomes are never used to fit the GNN.
* **Full batch on CPU.** `pyg-lib` has no wheel for this laptop's torch build, so `LinkNeighborLoader` cannot run here. R-GCN full batch takes about 1.3 s per epoch on CPU, so sampling is not needed at this graph size. HGT takes about 30 s per epoch. Open issue: the neighbour-sampled and cuGraph loader paths are not implemented or benchmarked; they need the GPU instance.
* **Two supervised regimes for baselines**, kept separate in every table: gene-grouped 5-fold CV on the 2020 to 2026 outcomes (measures signal, not deployable), and a prospective GBM fitted on 2018-01 to 2020-01 revisions (deployable at T).
* **CADD** comes from MyVariant.info, which reports no CADD version, so I cannot verify it predates T. It is reported only in rows labeled "version unverified" and is excluded from the primary tabular set. gnomAD is v2.1.1 (2019-03), confirmed from the MyVariant metadata endpoint.
* **Gene family stratification** uses current HGNC gene groups. Used only to stratify the report, never as a feature.
* GPU hours: 0. R-GCN wall time 17 min CPU for 5 folds.

Results (R-GCN; `results/m3/`):
* Time-T link prediction, held-out folds: P/LP vs B/LB AUROC 0.978 to 0.981; P/LP vs corrupted disease 0.985 to 0.992.
* Conformal (alpha 0.1): coverage 0.901 for both classes on held-out edges. 3,908 of 90,390 P/LP edges get the confident set {B/LB}.
* **H1, downgrade** (4,171 of 54,892): GNN alone 0.723; GBM tabular 0.837; tabular + GNN 0.839 (delta +0.002, CI -0.001 to +0.004, not significant); tabular + neighbour counts 0.847; adding GNN to that 0.845 (delta -0.002, significant, slightly worse). **Negative result: the GNN adds nothing beyond tabular metadata for downgrades.**
* **H1, upgrade** (2,131 of 73,438): GNN alone 0.819, equal to GBM tabular 0.820 without using outcome labels; tabular + GNN 0.881 (delta +0.061, CI 0.054 to 0.068); tabular + neighbour counts 0.894; adding GNN to that 0.896 (delta +0.002, CI -0.000 to +0.004, not significant). **Graph context helps for upgrades, but simple leave-one-out neighbour counts capture it as well as the GNN.**
* Prospective GBM (fitted on 2018 to 2020 revisions): 0.816 downgrade, 0.746 upgrade.

Failure: the first HGT run died when the previous session ended (no output). Restarted with a 24-epoch cap.

## M4. Variant dossier (QIAGEN) (20-variant set done; broader evaluation running)

Plan: per-rsID evidence gathering, reconciliation, cited plain text, automated verification, haplotype detection, CLI.

Design decisions:
* **Template-generated sentences, no LLM.** Each sentence is filled from values read out of a stored raw record and carries machine-checkable citations (record id, path, value). The verifier re-reads the record. This trades fluency for a guarantee that no sentence states a fact missing from its record.
* **Dated ClinVar from archived releases** (local SQLite index over 2018-01, 2019-01, 2020-01, 2026-11 variant summaries and the 2020-01 and 2026-11 submission summaries). `--as-of` uses only releases on or before the date; the assertion history is the sequence across releases.
* **Endpoints verified on 2026-11-16:** dbSNP refsnp API (merge history, alleles), Ensembl VEP REST, MyVariant.info (gnomAD v2.1.1), gnomAD GraphQL (v4.1, latest mode only), GWAS Catalog REST v2 (the legacy v1 API is retired), LitVar2, PubMed E-utilities, CIViC GraphQL.
* **as_of mode** drops gnomAD v4, CIViC, COSMIC and DoCM (no reliable release dates), filters GWAS and literature by PubMed publication date, and uses ClinVar releases up to the date. VEP and dbSNP are current releases used for identity and consequence only.
* **Haplotypes** are detected without gene-specific rules: same gene, within 100 kb, at least 20 shared LitVar2 PMIDs with Jaccard at least 0.10, then abstracts naming both rsIDs with haplotype language.
* **COSMIC counts are not openly accessible** without a licence; the dossier reports the COSMIC identifiers that Ensembl VEP exposes plus CIViC evidence counts and ClinVar oncogenicity classifications.

Results (`results/qiagen_20/`): 20 of 20 complete in both modes. Latest: 656 sentences, 1,828 citations, validity 1.0, no uncited sentence. As of 2020-01-01: 479 sentences, 1,216 citations, validity 1.0, no temporal violation. rs429358 + rs7412 detected as an APOE haplotype pair. rs80357713 resolves to rs80357783.

Bugs found by reading the output and fixed: ClinVar's placeholder "no classification for the single variant" was counted as a somatic signal (APOE was called a somatic mutation); GWAS p-values below 1e-308 printed as 0 (now built from mantissa and exponent); indel alleles failed to match ClinVar's left-aligned representation.

Limitations: the bottom line is rule-based; per-condition ClinVar classifications are not separated (aggregate per allele); literature examples are chosen by title match and recency, not relevance ranking.
