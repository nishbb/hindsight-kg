# NOTES

Running log of plans, decisions, failures and open issues. Newest milestone at the bottom.

## Compute log

| Milestone | Where | GPU hours | Running total GPU hours | Notes |
|---|---|---|---|---|
| M0 | laptop (Apple M-series, 11 CPU, 36 GB RAM) | 0 | 0 | |
| M1 | laptop | 0 | 0 | about 1 min CPU |
| M2 | laptop | 0 | 0 | 22 s CPU |
| M3 | laptop | 0 | 0 | R-GCN 17 min, HGT 42 min, evaluation about 40 min CPU |
| M4 | laptop | 0 | 0 | network-bound; about 1 h for 300 dossiers with a cold cache |
| M5 | laptop + NVIDIA NIM API | 0 | 0 | hosted LLM; tokens reported in results/m5 |
| M6 part 1 | laptop | 0 | 0 | 18 runs x 2.4 min CPU |
| M7 | laptop | 0 | 0 | a few minutes |

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
* 12 node types, 432,828 nodes: 353,600 variants (291,591 ClinVar, 62,009 GWAS-only), 21,008 genes, 21,838 diseases (MONDO), 14,586 phenotypes (HPO), 1,926 GWAS traits, plus GO terms, pathways, anatomy, compounds, symptoms.
* 32 edge types, 3,118,166 edges. Assertion edges at T: `assoc_plp` 90,390; `assoc_vus` 194,872; `assoc_blb` 74,216; `assoc_conflict` 33,239. GWAS variant-trait 89,603 (from 107,652 genome-wide significant rows added and published on or before T, of 166,103 raw rows).
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

## M3. GNN edge plausibility (done)

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

Failure: the first HGT run died when the previous session ended (no output). Restarted with a 24-epoch cap (42 min CPU for 5 folds).

HGT results: held-out P/LP vs B/LB AUROC 0.955 to 0.965, versus corrupted disease 0.857 to 0.877, and validation AUROC was still rising at the cap, so **HGT is undertrained here**; its lower H1 numbers (GNN alone 0.683 downgrade, 0.703 upgrade; +0.023 AUROC over tabular for upgrades, CI 0.017 to 0.029) should not be read as HGT being a worse architecture. A full HGT run needs the GPU instance. Conformal coverage for HGT: 0.905 (P/LP), 0.902 (B/LB). Combining both GNNs gives no gain over R-GCN alone.

## M4. Variant dossier (QIAGEN) (done)

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

### M4 broader evaluation (`results/m4_broad/summary.json`)

* 200 variants named in the brief (100 ClinVar expert-panel, 100 GWAS lead SNPs from the 2019-12-16 release, seeded sample from the pinned 2026-11 index): 200 of 200 dossiers complete, 4,660 sentences, 11,443 citations, citation validity 1.000, claim-to-record agreement 1.000, no uncited sentence.
* Conflict detection recall: 100 extra germline ClinVar variants whose aggregate classification is "Conflicting ..." (needed because expert-panel records essentially never carry the flag). Recall 1.0 both from the dossier's conflict list and when computed from individual submissions only. On the expert-panel set 31% of variants show submitter disagreement; these are real disagreements that the panel resolved, so they are reported as a disagreement rate, not as false alarms.
* Caveat: citation validity and agreement are 1.0 by construction where the template system works, so these numbers mainly show that the verifier finds nothing broken. They are not a test of whether the evidence is complete or well chosen.
* Bugs found by this run and fixed: a "no COSMIC record" sentence cited a field that is absent when Ensembl has no annotation for a very new rsID (3 variants); a transient Ensembl HTTP 500 aborted one dossier (now recorded as a processing note and the other sources continue).

## Pinning the outcome release (2026-11-17)

The first runs used `variant_summary.txt.gz` downloaded on 2026-11-16, which is NCBI's weekly update and is overwritten without being archived, so the labels could not be reproduced. All outcome labels now come from the archived monthly release `archive/variant_summary_2026-11.txt.gz` (Last-Modified 2026-11-05, md5 d7e0d7e581e220100fc84ca64fe953ba); the weekly file is kept in `data/raw/clinvar/weekly_2026-11-16/` only for the record. Label counts changed slightly (downgraded 4,171 to 4,170; stable VUS 71,307 to 71,361). The GNNs never see outcomes, so they were not retrained; every evaluation, the dossiers in latest mode, M5, M6 and M7 were regenerated. `scripts/download_data.py` pins every raw file by md5.

Final H1 numbers on the pinned release (`results/m3/h1_main.csv`, `h1_deltas.csv`): downgrade, tabular GBM 0.837 versus +R-GCN 0.839 (delta +0.002, CI -0.000 to +0.005); upgrade, tabular 0.812 versus +R-GCN 0.882 (delta +0.070), tabular + neighbour counts 0.895 versus +R-GCN 0.899 (delta +0.004, CI +0.002 to +0.006, now significant but small).

## M6. Does cleaning help? (part 1 done; part 2 not run)

Plan and decisions:
* **Prospective downstream task.** Train a gene-disease link predictor (R-GCN encoder, MLP decoder) on the time-T graph and test it on the 8,122 gene-disease pairs that have a germline P/LP ClinVar assertion in 2026-11 but none at T (diseases mapped through the same MONDO 2019 index). Negatives: 10 random (ClinVar gene, asserted disease) pairs per positive that are P/LP neither at T nor in 2026. Supervised training pairs are removed from the message graph each epoch.
* **Cleaning arms** (3,908 edges each, the size of the GNN flag): `gnn_flag` (out-of-fold R-GCN conformal set is {B/LB}; time-T only), `gbm_prosp` (top risk under the GBM fitted on 2018 to 2020 revisions), `random` (control), `oracle` (the 4,170 edges actually downgraded; uses future labels, upper bound only), `downweight` (flagged edges kept in a separate message relation; pairs supported only by flagged edges stay positives at loss weight 0.3). 3 seeds each.
* **Bug found and fixed:** the first `downweight` implementation also dropped flagged pairs from the training positives, so it was a second pruning arm. Fixed and all 18 runs regenerated.

Results (`results/m6/cleaning_summary.csv`, `cleaning_deltas.csv`), AUROC mean over 3 seeds: original 0.911; gnn_flag 0.889 (delta -0.023); downweight 0.906 (-0.005); gbm_prosp 0.905 (-0.006); random 0.908 (-0.004); oracle 0.908 (-0.003). **No arm improves downstream prediction; H3 is not supported.** Even removing exactly the edges that were later downgraded does not help, which suggests that for gene-disease discovery a downgraded variant-level assertion still carries useful signal about the gene-disease pair (the gene is usually still disease-relevant even when one variant is not). The GNN flag hurts most because it removes 1,625 gene-disease training pairs entirely.

Part 2 (STaRK-Prime graph-RAG with G-Retriever): **not run**. It needs a GPU for Llama-3.1-8B fine-tuning and none was attached. Preparation done (`scripts/m6_trust_transfer.py`): kg-gc trust can be transferred to only 5,326 of 80,411 PrimeKG disease-protein edges (6.6%), of which 958 have trust below 0.5. Any effect of trust-based pruning on STaRK-Prime is therefore limited to a small slice of the graph; we flag this as a reason to expect a small effect before spending GPU budget. Next step on the GPU instance: reproduce the G-Retriever baseline with PyG's `GRetriever` and QLoRA, then rerun with PCST edge cost increased by (1 - trust).

## M7. COPD / emphysema case study

See `docs/case_study_copd.md`. Seeds retrieved from the GWAS Catalog v2 API by trait-name search (3 MONDO terms, 1,234 associations, 562 lead loci). Only 12 time-T ClinVar edges qualify as COPD-related under the data rule; 3 were later downgraded (SCNN1A, SCNN1B bronchiectasis, all "no assertion criteria"), and all 3 had an "empty" conformal set at T. SERPINA1 drops out because its COPD GWAS support in the catalog is from 2024 (PMID 39024449), after T. The graph has no exposure entities; smoking-behaviour GWAS traits are the only exposure proxy.

Agent (config c, Nebius Nemotron-3_5-Lightning) on the 12 edges: 10 downgrade, 2 needs_review, accuracy 0.25. It flagged all 3 true downgrades and all 7 stable pathogenic edges, the same over-calling seen in M5.

## M5. Audit agent: provider problems and decisions (2026-11-17/18)

* First model choice on the 12-edge smoke set (excluded from evaluation): `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA NIM. During the full run it started returning HTTP 503 "Service temporarily overloaded".
* Switched to `nemotron-3.5-lightning` (same family, re-checked on the smoke set: config c 8 of 12 correct, valid verdicts, 4% unmatched citations). NIM's free tier then rate-limited the account (HTTP 429): with a shared client-side limiter throughput was 10 calls per minute, about 11 hours for the run.
* **Final provider: Nebius Token Factory** (OpenAI-compatible, `https://api.tokenfactory.nebius.com/v1/`, model `nvidia/Nemotron-3_5-Lightning`, key from `NEBIUS_API_KEY`), paid from the project's Nebius credits. No rate limiting observed (20 calls in 4.5 s). Third-party trackers list it at $0.06 per 1M input and $0.24 per 1M output tokens (not confirmed on Nebius's own price page, which needs a login); the full run is estimated at 20M input tokens, roughly $1.50 to $2.
* All LLM configs (b0, b, c, d) are run on this one model and provider; NIM responses are kept in the cache but not reported. Reasoning ("thinking") is switched off for all configs.
* Client hardening: 90 s request timeout with no hidden SDK retries (a stalled request previously blocked a run for 22 min), and a shared limiter that pauses all threads for 60 s after a 429 or 503.
* Known limitation: the LLM's weights were trained on data after T. Tool evidence is restricted to T, but the model may remember later reclassifications, which would favour the LLM configs. We cannot remove this; we report it.

## M8. Release checks (2026-11-18)

* `make reproduce-small` (labels, graph, small R-GCN, evaluation, three dossiers, two cleaning arms) ran end to end on the laptop CPU in 7 min 28 s wall clock, under the 30 min target. It also refreshed `results/m2/leakage_manifest.json`, whose `clinvar_latest` date was still the abandoned weekly file (2026-11-16); it now records the pinned archive (2026-11-05).
* 51 offline tests pass; ruff and black are clean.
