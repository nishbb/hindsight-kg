# Build prompt: hindsight-kg (predicting which genotype-phenotype knowledge graph edges will be revised)

Paste everything below the line into an agentic coding AI (Claude Code, Cursor agent, etc.) running in an empty Git repo with Python 3.11, internet access, and ideally one GPU.

---

## Role

You are a senior computational biologist and ML engineer building a research-grade, open-source GitHub repository. You work autonomously, in milestones, and you verify everything you build. You never fabricate data, results, API fields, or citations. When something fails, you debug it, and if you cannot fix it you write down exactly what failed and why in `NOTES.md` and move to the next milestone.

## Context

This project answers the NVIDIA problem statement from Nucleate Pittsburgh BioHack 2026: "How can we prototype agentic frameworks that leverage GNNs to parse phenotype-genotype knowledge graphs?" It also includes a module that answers the QIAGEN problem statement: "Given a list of SNPs, mine and reconcile data from publicly available datasets, then integrate findings with primary literature to produce a clear, plain-text summary of each variant's biological function, frequency, and disease relevance."

Open problems named by the NVIDIA genomics team for biomedical knowledge graphs include: pruning low-quality data, validation, garbage collection, dynamic data storage, and handling multifactorial disease. This project targets pruning, validation and garbage collection directly.

## Core research question and novelty

Genotype-phenotype knowledge graphs (PrimeKG, Hetionet, commercial KGs) treat every edge as equally true, but variant-disease assertions get revised over time. ClinVar publishes monthly archived releases, so we can travel back in time, build a KG from an old snapshot, and see which edges were later downgraded (e.g. Pathogenic/Likely pathogenic to VUS/Benign) or upgraded (VUS to Pathogenic/Likely pathogenic). This gives a real, prospective ground truth for "which KG edges are garbage" without any manual labeling.

Hypotheses to test:

- H1: A heterogeneous GNN trained on a time-T knowledge graph assigns lower plausibility to variant-disease edges that are later downgraded, and higher plausibility to non-edges (VUS) that are later upgraded, better than simple baselines.
- H2: An LLM agent that combines GNN scores and explanations with live evidence retrieval (restricted to evidence available at time T for evaluation) makes better keep/downgrade/needs-review decisions than either the GNN alone or the LLM alone.
- H3: Pruning low-trust edges improves downstream link prediction or graph-RAG question answering on the cleaned KG.

Before writing any code, do a focused related-work check (search arXiv, bioRxiv, PubMed for: ClinVar reclassification prediction, knowledge graph error detection, knowledge graph refinement biomedical, TxGNN, SHEPHERD, G-Retriever, Biomni). Write `docs/related_work.md` with real, verified references only (title, authors, year, link) and a short paragraph stating precisely what is new here relative to each. If you find prior work that already does H1 to H3 together, stop and report this in `NOTES.md` with a proposal for how to differentiate.

Also check these specifically, since they come from the same community that set this problem: the `collaborativebioinformatics/BioGraphRAG` GitHub repo, `neo4j-product-examples/neo4j-gnn-llm-example`, the STaRK benchmark paper (STaRK-Prime), and the NVIDIA developer blog post "Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases".

## Compute budget

Target hardware is a single NVIDIA Brev GPU instance in the RTX 6000 class (roughly 48 GB VRAM), with a limited credit budget. Nebius credits are available for CPU-heavy work. Design accordingly:
- Do all downloading, parsing, ID mapping, and graph construction on CPU (laptop or Nebius CPU instance). Only GNN training and LLM fine-tuning touch the GPU.
- Save graph artifacts to disk so the GPU instance only loads prebuilt tensors.
- Use mini-batch neighbor sampling for the GNN and QLoRA (4-bit) for any LLM fine-tuning so everything fits on one 48 GB card.
- Log GPU hours per milestone in `NOTES.md` and keep a running total. Before any GPU job expected to exceed 2 hours, estimate its cost and run a 5-minute smoke test first.
- LLM agent inference should go through a hosted API (NVIDIA NIM at build.nvidia.com or similar), not a self-hosted model, unless a milestone explicitly needs fine-tuning.

## Hard rules

1. Public data only. Never download or reference controlled-access cohort data (e.g. COPDGene or SPIROMICS individual-level data, dbGaP).
2. No temporal leakage. Every feature used to score an edge at time T must have existed at time T. Track the release date of every source. gnomAD v4 (2023), AlphaMissense (2023), and anything trained on later ClinVar labels are leaky for a 2019/2020 snapshot. Either exclude them from the temporal experiment or report them as a separately labeled "leaky upper bound".
3. Every factual sentence in any generated variant summary must carry a citation to a retrieved record (ClinVar VCV/RCV ID, gnomAD version, PubMed ID, GWAS Catalog accession). Run an automated verification pass that checks each cited claim against the stored raw record.
4. Cache every API response to disk (`data/cache/`) with timestamp and source version, respect rate limits, and make all runs reproducible from a single command with fixed seeds.
5. Report negative results honestly. If the GNN does not beat a baseline, say so in the README.
6. Do not use em dashes in any prose you write (README, docs, summaries).

## Data sources (verify each endpoint before relying on it)

- ClinVar archived monthly `variant_summary.txt.gz` from the NCBI FTP archive (`ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/`). Pick a past snapshot T (target: early 2020) and a recent snapshot (latest available).
- STaRK-Prime (the PrimeKG-based split of the STaRK retrieval benchmark) for graph-RAG question answering evaluation in M6.
- PrimeKG (Harvard Dataverse) as the base gene-phenotype-disease-drug graph. It uses MONDO disease IDs; map ClinVar MedGen/OMIM conditions to MONDO using MONDO cross-references.
- HPO gene-phenotype annotations (dated release close to T for the temporal experiment).
- GWAS Catalog (REST API or dated downloads; association records carry publication dates, so filter to before T).
- Variant annotation: Ensembl VEP REST API, MyVariant.info, gnomAD (version available at T, e.g. v2.1.1 for 2020), CADD.
- Literature: NCBI LitVar2 (rsID to PubMed), PubTator3, PubMed E-utilities. For temporal evaluation, filter papers to publication date before T.
- Optional: Open Targets Platform GraphQL API.

## Architecture

```
hindsight-kg/
  README.md                 # problem, method diagram, results tables, how to reproduce
  NOTES.md                  # running log of decisions, failures, open issues
  pyproject.toml / environment.yml
  configs/                  # yaml configs per experiment
  src/hindsight/
    data/                   # downloaders, parsers, snapshot builders, ID mapping
    graph/                  # hetero graph construction (PyG HeteroData), temporal splits
    models/                 # R-GCN and HGT link predictors, calibration (conformal)
    explain/                # path-based explanations (GNNExplainer or metapath scoring)
    variant_dossier/        # QIAGEN module: per-SNP evidence mining and summaries
    agent/                  # tool-using LLM agent: tools, prompts, audit loop
    eval/                   # metrics, baselines, ablations, plots
  scripts/                  # one entry point per milestone
  tests/                    # pytest unit tests for parsers, mappings, leakage checks
  docs/related_work.md
  results/                  # small result tables and figures committed to git
```

LLM backend: use an OpenAI-compatible client with configurable base URL and model so the agent can run on NVIDIA NIM endpoints (build.nvidia.com, e.g. a Nemotron or Llama model) or any other provider. Read API keys from environment variables only. Use PyTorch Geometric for the GNN; if cuGraph-PyG / WholeGraph is available on the machine, add an optional accelerated data loader path and benchmark it against plain PyG.

## Milestones (complete in order, commit after each, with passing tests)

### M0. Setup
Environment file, package skeleton, pre-commit (ruff, black), pytest, GitHub Actions CI running tests on CPU with a tiny fixture graph. MIT license. `docs/related_work.md` done.

### M1. Temporal ClinVar labels
Download snapshot T and latest. Keep germline, single-condition-resolvable records with a clear clinical significance. Build the label table:
- `downgraded`: P/LP at T, VUS/B/LB/conflicting at latest
- `stable_pathogenic`: P/LP at both
- `upgraded`: VUS at T, P/LP at latest
- `stable_vus`: VUS at both
Report counts, distribution by gene, by review status, and by submitter count. Unit test: no variant ID appears in two classes.

### M2. Time-T knowledge graph
PrimeKG plus variant nodes from ClinVar at T (variant to gene, variant to disease with significance as edge type), HPO annotations, GWAS Catalog variant-trait edges published before T. Save as PyG `HeteroData`. Write a leakage test that asserts no edge, node feature, or literature item post-dates T.

### M3. GNN edge plausibility
Train R-GCN and HGT link predictors on the time-T graph with the variant-disease pathogenic edges as the target relation (proper train/val/test edge splits, negative sampling, early stopping). Output a plausibility score for every existing P/LP edge and every VUS non-edge. Add split conformal calibration so scores map to calibrated confidence sets.

Baselines (all computed with time-T information only): ClinVar review stars, number of submitters, presence of conflicting submissions history at T, gnomAD AF at T, CADD, and a gradient-boosted model over these tabular features.

Evaluation: AUROC and AUPRC for predicting `downgraded` vs `stable_pathogenic`, and `upgraded` vs `stable_vus`. Bootstrap 95% CIs. Stratify results by gene family and review status. Also run a model that combines GNN score with tabular features to test whether the graph adds signal beyond the tabular baseline. This comparison is the key H1 result.

### M4. Variant dossier module (QIAGEN problem)
CLI: `snp-dossier --rsids rs1,rs2 --as-of 2020-01-01|latest --out out/`.
For each SNP, gather: gene and consequence (VEP), allele frequency overall and per ancestry (gnomAD), clinical assertions and their history (ClinVar), trait associations (GWAS Catalog), literature (LitVar2 then abstracts). Reconcile: detect and explicitly report conflicts (e.g. pathogenic claim while AF exceeds an ACMG BA1-like threshold, conflicting ClinVar submitters, discordant gene assignments). Output a structured JSON and a plain-text summary of function, frequency, disease relevance and conflicts, every sentence cited.
Required test set (the official QIAGEN hackathon list; these 20 must all produce a complete dossier and be committed to `results/qiagen_20/` as both JSON and plain text, plus one combined `qiagen_20_report.md`):

```
rs9807334, rs121913529, rs72921001, rs17822931, rs334,
rs987870, rs429358, rs3827760, rs113993960, rs4988235,
rs28934578, rs17004038, rs12913832, rs6025, rs713598,
rs7412, rs80357713, rs1426654, rs1799963, rs1801133
```

This list is deliberately heterogeneous. Do not assume every entry is a simple germline SNV. For each rsID, resolve and handle:
- Variant type: some are indels or multi-nucleotide variants, not SNPs. Report the true type and HGVS notation.
- Multi-allelic sites: report each alternate allele separately with its own frequency and clinical meaning, rather than merging them.
- Germline vs somatic relevance: some are mainly cancer somatic hotspots. Say so, and pull somatic evidence (e.g. COSMIC counts if openly accessible, or CIViC) separately from germline ClinVar evidence.
- Haplotypes: if two rsIDs in the list jointly define a known haplotype (e.g. APOE alleles), detect this automatically from the literature and annotation data and add a joint interpretation section, in addition to the per-SNP entries.
- Population differentiation: for trait and ancestry-informative variants, frequencies differ sharply by ancestry. Always report per-ancestry gnomAD frequencies and flag large differences.
- Trait vs disease: for non-disease traits (pigmentation, taste, metabolism), describe the trait association and state clearly when there is no established disease relevance.
- Merged or retired rsIDs: check dbSNP merge history and report the current ID.
Each dossier should end with a one-line "bottom line" in plain language for a non-specialist, plus a confidence level tied to the strength of the evidence.

Broader evaluation: on 200 additional variants (100 ClinVar expert-panel reviewed, 100 GWAS lead SNPs) measure citation validity rate, automated claim-to-record agreement, and conflict detection recall against ClinVar conflicting-interpretation flags.
The dossier functions double as the agent's evidence tools in M5.

### M5. GNN-guided audit agent (NVIDIA problem)
Tools exposed to the agent: `kg_neighborhood(node, hops)`, `gnn_score(edge)`, `gnn_explain(edge)` returning top supporting and contradicting paths, `dossier(variant, as_of)`, and `literature_search(query, as_of)`.
Loop: take the lowest-trust P/LP edges and highest-scoring VUS non-edges from M3, have the agent investigate each and output a structured verdict (`keep`, `downgrade`, `upgrade`, `needs_review`), a confidence, and a cited rationale. The agent should prioritize which edges to spend tool calls on using the conformal confidence (spend more on uncertain edges).
Evaluate on a held-out sample (at least 300 edges) against the latest ClinVar outcome, with evidence restricted to time T. Ablations: (a) GNN only, (b) LLM agent without GNN tools, (c) agent with GNN tools, (d) agent with GNN tools plus conformal budget allocation. Report accuracy, macro-F1, calibration (ECE), tool calls per edge, and token cost.

### M6. Does cleaning help? (H3)
Two evaluations:
1. Prune or down-weight edges flagged by the best method from M5 and retrain link prediction for disease-gene edges on the cleaned vs original graph. Report the change in held-out performance.
2. STaRK-Prime graph-RAG: reproduce a G-Retriever baseline on STaRK-Prime using PyG (NVIDIA reports about 32% hits@1 with a GAT plus Llama-3.1-8B-Instruct pipeline; reproduce rather than quote, and report your own number). Then rerun with the KG cleaned by hindsight-kg trust scores, either pruning low-trust edges or adding trust as an edge feature or as a Prize-Collecting Steiner Tree edge cost. Report hits@1, hits@5, and MRR for original vs cleaned. Use QLoRA so it fits on one RTX 6000 class GPU. If full fine-tuning exceeds the budget, use a subsample and say so.

### M7. Multifactorial case study: COPD / emphysema
Seed from GWAS Catalog COPD and emphysema associations (retrieve, do not hardcode). Run the dossier on lead SNPs and the audit agent on COPD-related variant-disease edges. Produce a subgraph figure linking variants, genes, proteins, phenotypes, and exposures where the KG supports it, highlighting edges whose trust changed. Write a short narrative in `docs/case_study_copd.md`.

### M8. Write-up and release
README with: one-paragraph abstract, method figure, results tables with CIs, ablations, limitations, reproduction commands, data licenses. Add `paper/` with a 4 to 6 page workshop-style draft in Markdown or LaTeX (intro, related work, methods, results, limitations). Add a `CITATION.cff`. Tag release v0.1.0.

## Definition of done

- `make reproduce-small` runs end to end on CPU in under 30 minutes on a subsampled graph.
- All tests pass in CI, including the leakage tests.
- Every number in the README comes from a file in `results/` generated by a script.
- `docs/related_work.md` and `NOTES.md` are complete and honest.

## Working style

Plan each milestone in `NOTES.md` before coding. Prefer small, verified steps. After each milestone, print a short status: what was built, key numbers, what failed, what is next. If a design decision has two reasonable options and the choice materially changes results, log both and pick the more conservative one.
