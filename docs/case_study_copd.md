# Case study: COPD and emphysema

Generated inputs: `scripts/m7_copd.py`. Numbers below come from `results/m7/summary.json`, `results/m7/copd_edges.csv`, `results/m7/lead_snps.csv` and `results/m7/dossiers/`. The figure is `results/m7/copd_subgraph.png`.

## Why COPD is a hard case for a genotype-phenotype knowledge graph

COPD is multifactorial. Most of the heritable signal sits in many common variants of small effect, the dominant exposure (tobacco smoke) is environmental, and the one well-established monogenic route (alpha-1 antitrypsin deficiency, SERPINA1) explains a small fraction of cases. A knowledge graph built around ClinVar's variant-disease assertions sees almost none of this: ClinVar is designed for Mendelian assertions, so "COPD" barely appears in it.

## What we retrieved

Seeds were retrieved, not hardcoded. A GWAS Catalog v2 trait search for "chronic obstructive pulmonary disease" and "emphysema" returned three disease terms (MONDO_0005002 chronic obstructive pulmonary disease, MONDO_0004849 pulmonary emphysema, MONDO_0000923 interstitial emphysema). Their associations: 1,234 records with an rsID, 1,002 genome-wide significant, 562 distinct lead loci after collapsing by mapped gene.

The strongest loci (lowest p-value per locus) were NPNT (rs34712979), HTR4 (rs10037493), ADGRG6 (rs7753012), THSD4 (rs2165489), CHRNA3 (rs12914385), AGER/PBX2 (rs9391855), CDC123 (rs2001546) and CHRNA5/PSMA4 (rs2036527), all in line with the published COPD and lung-function GWAS literature. Dossiers for the 12 strongest lead SNPs are in `results/m7/dossiers/`; every one is classified by the dossier as a "trait-associated common variant without an established monogenic disease role", with citation validity 1.0. Two of them (CHRNA3 rs12914385 and CHRNA5/PSMA4 rs2036527) sit in the nicotinic receptor cluster, whose strongest associations are with smoking behaviour and lung cancer, which is the clearest example of gene-exposure entanglement in this set.

## How much of COPD the time-T graph can audit

Edges were selected with a data rule (see the script docstring): ClinVar assertions at T whose disease lies in the MONDO 2019 subtree of the three traits, plus respiratory-system diseases whose gene is mapped by a genome-wide significant COPD or emphysema association **published before 2020**. Only **12 edges** qualify:

* 9 in the MONDO COPD subtree, all bronchiectasis with or without elevated sweat chloride (SCNN1A, SCNN1B), 7 of them "no assertion criteria provided";
* 3 for diaphragmatic hernia 3 (ZFPM2), included because ZFPM2 is a pre-2020 COPD GWAS gene and the disease is a respiratory-system disease in MONDO.

**SERPINA1 is absent, and that is correct under the time-T rule.** Alpha-1 antitrypsin deficiency is a respiratory-system disease in MONDO, but the SERPINA1 COPD associations in the current GWAS Catalog (for example rs28929474, the Z allele) come from a 2024 study (PMID 39024449). Using them to select 2020 edges would leak post-T knowledge. This is a concrete instance of the general problem: the evidence that ties a gene to a common disease often arrives years after the gene's Mendelian assertions.

## What happened to these edges by 2026

Of the 12 edges, 3 P/LP assertions were downgraded to uncertain significance by 2026-11, all bronchiectasis variants submitted without assertion criteria: SCNN1B p.Pro369Thr (VCV 8842), SCNN1B p.Asn288Ser (VCV 8843) and SCNN1A p.Val114Ile (VCV 9268). 7 P/LP assertions stayed pathogenic and 2 VUS stayed VUS.

At T the R-GCN conformal set was "empty" (the edge resembles neither a typical pathogenic nor a typical benign assertion) for **all 3 downgraded edges** and for 4 of the 7 that stayed pathogenic. The 3 edges with a confident pathogenic-like set all stayed pathogenic. With 12 edges this is anecdote, not evidence, but it matches the pattern in the full evaluation: the GNN separates weakly supported assertions from strongly supported ones, and it cannot tell which weakly supported ones will actually be revised.

## What the knowledge graph supports, and what it does not

The figure links GWAS lead variants (time-T graph, GWAS Catalog 2019-12-16 release) to lung-function traits (FEV1/FVC, diffusing capacity, emphysema imaging), to smoking-behaviour traits (the only "exposure" information the graph has, as GWAS traits), to genes, to HPO respiratory phenotypes of those genes, and to the 12 ClinVar edges, with the three revised assertions drawn as red dashed lines. The graph does **not** contain:

* exposures as first-class nodes (no smoking, air pollution or occupational exposure entities);
* gene-environment interactions or effect sizes;
* any link from the common-variant COPD loci to the Mendelian bronchiectasis genes beyond shared HPO phenotypes.

## Take-aways for KG garbage collection in multifactorial disease

1. Edge-level trust from ClinVar history covers only the Mendelian fringe of COPD (12 edges). Most of the COPD knowledge is in GWAS associations, whose revision history is a different problem (replication and fine-mapping, not reclassification).
2. Time-T discipline changes the case study materially: the best-known COPD gene (SERPINA1) drops out because its GWAS support postdates T.
3. The audited edges that were revised were the low-review, OMIM-era single-submitter assertions, which is the same signal that dominates the tabular baselines in the full evaluation.

## Audit agent on the 12 COPD edges

The audit agent (config c: GNN tools, dossier as of 2020-01-01, PubMed before 2020; Nemotron 3.5 Lightning, reasoning off) was run on all 12 edges. It returned `downgrade` for 10 and `needs_review` for the 2 uncertain edges, so it was correct on 3 of 12 (accuracy 0.25): the three true downgrades were caught, but so were 7 of the 7 stable pathogenic edges. This repeats the failure seen in the full M5 evaluation (precision for downgrade at chance), where the agent treats weak documentation (no assertion criteria, single submitter) as evidence of error. The GNN conformal set alone separated the cases better: all 3 downgraded edges had an empty set, while 3 of the 7 stable pathogenic edges (ZFPM2 VCV 6129 and 6131, SCNN1A VCV 504974) had a confident P/LP set. Verdicts and rationales are in `results/m7/agent_verdicts.csv`.
