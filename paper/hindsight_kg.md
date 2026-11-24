# Hindsight: Using ClinVar's Revision History to Test Knowledge-Graph Trust Scoring, GNN-Guided Audit Agents and Graph Cleaning

*Workshop draft. All numbers are produced by scripts in the accompanying repository and copied from `results/tables.md`.*

## Abstract

Genotype-phenotype knowledge graphs (KGs) treat every edge as equally true, yet variant-disease assertions are routinely revised. We use ClinVar's monthly archives as a prospective, label-free ground truth: we build a heterogeneous KG exactly as it stood in January 2020 and ask which of its variant-disease edges were downgraded or upgraded by November 2026. Under a strict temporal-leakage protocol we test three hypotheses. (H1) A relational GCN trained on the 2020 graph without outcome labels ranks later-upgraded uncertain variants well (AUROC 0.819) and adds 0.070 AUROC to a tabular model of ClinVar metadata; for downgrades it adds nothing, and simple leave-one-out neighbour counts capture almost all graph signal in both tasks. (H2) An LLM agent with GNN, dossier and literature tools restricted to 2020 evidence scores significantly below the GNN's conformal decision rule (macro-F1 0.30 to 0.37 versus 0.52), although that rule is itself weak: its accuracy is not significantly above always answering "keep" (0.550 versus 0.500) and it catches 6 of 80 downgrades. Allocating tool calls by conformal uncertainty gives a suggestive gain (+0.056 macro-F1, uncorrected). (H3) Pruning or down-weighting low-trust edges does not improve prospective gene-disease link prediction, even when the pruned edges are exactly those later downgraded. We also release a variant dossier generator whose every sentence is mechanically verified against a stored source record (100% citation validity on 220 variants). The negative results bound what graph-learning "garbage collection" can currently deliver for clinical-variant KGs.

## 1. Introduction

Biomedical KGs such as Hetionet, PrimeKG and commercial graphs integrate gene-disease, variant-disease and phenotype assertions from curated databases and treat them as facts. NVIDIA's genomics team lists pruning low-quality data, validation and garbage collection among the open problems for such graphs. Most work on KG error detection evaluates on synthetically corrupted triples, which may not resemble real errors. ClinVar offers something better: every month it publishes a full snapshot, and roughly 8% of pathogenic assertions in a 2020 snapshot were downgraded by 2026. These revisions are expert decisions made after new evidence, which makes them a natural prospective label for "edges that should not have been trusted".

We ask three questions. Does graph structure, learned by a GNN without any outcome labels, identify the edges that were later revised better than the metadata curators already see (H1)? Does an LLM agent that combines GNN scores with time-restricted evidence retrieval audit edges better than the GNN alone (H2)? And does removing low-trust edges improve downstream prediction (H3)?

## 2. Related work

Tabular models of ClinVar reclassification (RENOVO, Favalli et al. 2021; its 2024 follow-up; a 2025 leakage-controlled thesis) show that metadata predicts revision but use no graph. PathoGN (Kamada et al. 2021) evaluates a GCN on one year of ClinVar changes. KG error detection methods (KGTtm, CCA, MAKGED) and the hackathon project Model Garbage Collection detect synthetic or simulated errors. Graph-RAG systems (G-Retriever; the NVIDIA/Neo4j STaRK-Prime pipeline; BioGraphRAG) assume the graph is correct. TxGNN and SHEPHERD use PrimeKG-style graphs as fixed truth. Full references and a per-paper comparison are in `docs/related_work.md`. To our knowledge no prior work tests H1 to H3 together on real, prospective revisions.

## 3. Data and temporal protocol

**Snapshot T.** ClinVar `variant_summary` 2020-01 (FTP timestamp 2020-01-02). Every input to the time-T graph must be dated on or before this: Hetionet v1.0 (2017) as the molecular background (genes, GO, pathways, anatomy, compounds, protein interactions); MONDO 2019-11-25 for diseases and the mapping of ClinVar conditions (only `equivalentTo` cross-references; ambiguous mappings dropped); HPO 2019-11-08 and its 2019-09 gene-phenotype annotations (recovered from the Internet Archive; no complete pre-2020 disease-phenotype file exists); GWAS Catalog release 2019-12-16 filtered on date added and publication date. PrimeKG is built from 2021 sources and is used only where flagged as leaky.

**Graph.** 12 node types and 432,828 nodes (353,600 variants, 21,008 genes, 21,838 MONDO diseases, 14,586 HPO phenotypes), 32 edge types and 3.1M edges. Variant-disease assertions are typed edges (pathogenic, uncertain, benign, conflicting). Variant node features are intrinsic only (HGVS-derived consequence and variant type).

**Labels.** Germline records, exactly pathogenic/likely pathogenic or uncertain at T, a single condition mapped to one MONDO term and a single gene. The outcome is the aggregate classification of the same VariationID in the archived 2026-11 release: 4,170 downgraded versus 50,722 stable pathogenic; 2,131 upgraded versus 71,361 stable uncertain.

**Leakage tests.** Every node and edge type records its source; automated tests fail on any source released after T or flagged leaky, on any post-T ClinVar or GWAS date, and on any feature outside a time-T allow list. CADD is reported separately because its version cannot be verified as pre-T.

## 4. Methods

**GNN plausibility.** R-GCN (per-relation mean aggregation, residual, 2 layers, 64 dimensions) and HGT encoders with an MLP decoder over (variant, disease) pairs. Training separates held-out pathogenic edges from benign edges and random-disease corruptions, using time-T classes only. Variants are split into five folds; each fold's assertions are absent from both the message graph and supervision of the model that scores them. Assertion edges point only from variant to disease, so a variant never receives a message encoding its own class. Split conformal prediction (class-conditional, alpha 0.1) on held-out validation edges converts scores to prediction sets.

**Baselines.** Single features (review stars, submitter count, prior conflict, gnomAD v2.1.1 frequency, CADD); gradient-boosted trees over all time-T metadata, evaluated with gene-grouped cross-validation on the outcome; the same with leave-one-out neighbour counts (other variants' classes in the same gene, gene and disease, gene and consequence); and a prospective model trained only on 2018 to 2020 revisions.

**Audit agent.** A tool-calling LLM (Nemotron 3.5 Lightning, reasoning off, via Nebius Token Factory) audits one edge and returns keep, downgrade, upgrade or needs_review, a confidence and a rationale citing evidence IDs. Tools: graph neighbourhood, GNN score with conformal set, metapath explanation (supporting and contradicting paths), a variant dossier as of 2020-01-01, and PubMed search before 2020. Ablations: (a) GNN conformal rule; LLM without tools; (b) agent without GNN tools; (c) with GNN tools; (d) with GNN tools and a budget of 3 calls for confident conformal sets and 20 otherwise (mean 6, matched to c). Evaluation on 320 edges, 80 per outcome, disjoint from the 12 edges used to choose the model and prompt.

**Cleaning.** A gene-disease link predictor trained on the time-T graph is tested on the 8,122 gene-disease pairs that first gain a pathogenic assertion by 2026. Arms remove 3,908 edges each (GNN flag, prospective GBM, random), down-weight the GNN-flagged edges, or remove the 4,170 edges truly downgraded (oracle). Three seeds.

**Variant dossier.** Templates fill sentences from stored records of dbSNP, Ensembl VEP, gnomAD, archived ClinVar releases and submissions, CIViC, GWAS Catalog and PubMed; each sentence carries citations (record, path, value) that a verifier re-checks. Rules detect frequency-versus-pathogenicity conflicts, submitter disagreement, allele-specific meaning, gene-assignment differences and haplotype pairs (literature co-occurrence plus abstracts).

## 5. Results

**H1.** Time-T link prediction is strong (pathogenic versus benign AUROC 0.978 to 0.981 across folds) and conformal coverage is nominal (0.901). For downgrades, the R-GCN alone reaches 0.723 AUROC versus 0.837 for tabular metadata, and adding it to the tabular model changes AUROC by +0.002 (95% CI -0.000 to +0.005). For upgrades the R-GCN alone (0.819) matches the supervised tabular model (0.812), and adding it raises AUROC to 0.882 (+0.070). Leave-one-out neighbour counts give 0.895, and adding the R-GCN on top gives +0.004. HGT, capped at 24 CPU epochs, is weaker throughout. The prospective model, which could have been deployed in 2020, reaches 0.816 and 0.746.

**H2.** The GNN conformal rule (accuracy 0.550, macro-F1 0.516) scores above all agent configurations (0.294 to 0.328; 0.303 to 0.372), all p < 0.001. Always answering "keep" scores accuracy 0.500 and macro-F1 0.222 on this sample, so the GNN rule's accuracy gain over it is not significant (+0.050, 95% CI -0.022 to +0.119); its macro-F1 gain is (+0.294, CI +0.239 to +0.343) and comes mainly from upgrades (62 of 80 caught, 21 of 80 stable VUS wrongly upgraded). It caught only 6 of 80 downgrades. Agents downgrade nearly every pathogenic edge (precision 0.50 to 0.51, chance level) and rarely upgrade. GNN tools alone do not help (b to c, p = 0.64); conformal budget allocation does (c to d, +0.056 macro-F1, p = 0.017). It also used 65% more prompt tokens, and it is one of 18 uncorrected comparisons, so we treat it as suggestive. Citations to evidence never returned by a tool fell from 55% to about 1% after we removed a GNN-specific example from the shared prompt; we report only the corrected run.

**H3.** Original graph AUROC 0.911. Pruning GNN-flagged edges: 0.889; down-weighting them: 0.906; prospective GBM: 0.905; random: 0.908; oracle: 0.908. No arm helps.

**Dossier.** 20 of 20 QIAGEN variants produce complete dossiers (656 cited sentences, 1,828 citations, validity 1.000; 479 sentences in 2020 mode with no temporal violation); 200 of 200 on 100 expert-panel and 100 GWAS lead variants; conflict recall 1.00 against ClinVar's conflicting flag.

## 6. Discussion

The headline is negative and, we think, informative. Revision of pathogenic assertions in ClinVar is predicted well by a tabular model whose largest feature is missense consequence, followed by review stars, gnomAD frequency and likely versus definite status; graph structure adds little once that is known. Graph context does help to find uncertain variants that will become pathogenic, because those sit in genes and diseases with many other pathogenic variants, but explicit counts of those neighbours do the job as well as a GNN. An LLM agent with access to the same evidence scores below the conformal rule because it over-calls downgrades; we have not tested why. The rule is weak for downgrades too, so the H2 comparison does not show that the GNN solves edge auditing. Cleaning did not help, possibly because a downgraded variant-level assertion can still carry correct gene-level information; we did not test that mechanism. Our results argue against deleting edges by these trust scores. Using them to prioritise human review is an untested alternative.

## 7. Limitations

The outcome is ClinVar's later opinion, not biological truth. Labels use variant-level aggregate classifications. Disease-phenotype annotations before 2020 are unavailable in complete form. HGT is undertrained and no GPU was used. The agent uses one small model with reasoning off; LLM pretraining postdates T and can leak later knowledge, which would favour the agent (and it still loses). The STaRK-Prime graph-RAG evaluation was not run; our trust transfer covers only 6.6% of PrimeKG disease-protein edges, which bounds any effect there.

## 8. Reproducibility

Pinned, checksummed inputs (`scripts/download_data.py`), cached API responses, fixed seeds, `make reproduce` and `make reproduce-small`, and a running decision log (`NOTES.md`).
