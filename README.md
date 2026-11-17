# hindsight-kg

**Hindsight: predicting which genotype-phenotype knowledge graph edges will be revised, using GNN-guided agents.**

Status: early development. No results yet.

## The idea

Genotype-phenotype knowledge graphs treat every edge as equally true, but variant-disease assertions get revised. ClinVar publishes monthly archived releases, so we can rebuild a knowledge graph as it looked at a past date, then check which edges were later downgraded or upgraded. That gives a prospective, label-free ground truth for knowledge graph quality.

On top of that ground truth, this project tests:

1. Whether a heterogeneous GNN trained on the past graph can predict which edges will be revised, beyond simple baselines (review status, allele frequency, in silico scores).
2. Whether an LLM agent that uses GNN scores and explanations as tools, plus evidence retrieval restricted to the past date, audits edges better than the GNN or the LLM alone.
3. Whether cleaning the graph with these trust scores improves downstream tasks, including graph-RAG question answering on STaRK-Prime.

The repo also includes `snp-dossier`, a tool that mines and reconciles public data and literature for a list of SNPs and writes a cited, plain-text summary of each variant's function, frequency, and disease relevance.

Built in response to the NVIDIA and QIAGEN BioTech track problem statements at Nucleate Pittsburgh BioHack 2026.

## Plan

See [`docs/BUILD_PROMPT.md`](docs/BUILD_PROMPT.md) for the full research plan, hard rules, and milestones.

## Data

Public data only: ClinVar archives, PrimeKG, HPO, GWAS Catalog, gnomAD, Ensembl VEP, MyVariant.info, NCBI LitVar2 and PubTator3, STaRK-Prime. No controlled-access cohort data.

## License

MIT
