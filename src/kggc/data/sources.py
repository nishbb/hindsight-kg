"""Registry of every data source, its release date, and where it is used.

The leakage tests read this registry: any source used in the time-T graph or time-T features
must have ``release_date < T`` unless it is explicitly flagged ``leaky`` and only used in a
separately labeled "leaky upper bound" analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

T_SNAPSHOT = "2020-01"
T_DATE = "2020-01-02"  # FTP timestamp of variant_summary_2020-01.txt.gz; time-T inputs must be dated <= T_DATE


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    version: str
    release_date: str  # ISO date of release (or of the archive capture for mirrored files)
    url: str
    license: str
    used_for: str
    leaky: bool = False
    note: str = ""


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            "clinvar_T",
            "ClinVar variant_summary",
            "2020-01",
            "2020-01-02",
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2020/variant_summary_2020-01.txt.gz",
            "Public domain (NCBI)",
            "time-T variant nodes, assertion edges, tabular features",
            note="FTP Last-Modified 2020-01-02 05:05 GMT.",
        ),
        Source(
            "clinvar_2019",
            "ClinVar variant_summary",
            "2019-01",
            "2019-01-03",
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2019/variant_summary_2019-01.txt.gz",
            "Public domain (NCBI)",
            "pre-T assertion history features",
        ),
        Source(
            "clinvar_2018",
            "ClinVar variant_summary",
            "2018-01",
            "2018-01-04",
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2018/variant_summary_2018-01.txt.gz",
            "Public domain (NCBI)",
            "pre-T assertion history features",
        ),
        Source(
            "clinvar_latest",
            "ClinVar variant_summary",
            "2026-11 monthly archive (Last-Modified 2026-11-05)",
            "2026-11-05",
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-11.txt.gz",
            "Public domain (NCBI)",
            "OUTCOME LABELS ONLY (never a feature)",
            leaky=True,
        ),
        Source(
            "hetionet",
            "Hetionet",
            "v1.0",
            "2017-07-31",
            "https://github.com/hetio/hetionet",
            "CC0 1.0 (hetnet); some underlying sources carry their own terms",
            "time-safe molecular background (genes, GO, pathways, anatomy, compounds, PPI)",
            note="Built from source releases up to 2016-2017 (Himmelstein et al. 2017).",
        ),
        Source(
            "mondo_T",
            "MONDO disease ontology",
            "v2019-11-25",
            "2019-11-25",
            "https://github.com/monarch-initiative/mondo/releases/tag/v2019-11-25",
            "CC BY 4.0",
            "disease nodes, disease hierarchy, ClinVar condition to MONDO mapping",
        ),
        Source(
            "hpo_obo_T",
            "Human Phenotype Ontology",
            "v2019-11-08",
            "2019-11-08",
            "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/v2019-11-08/hp.obo",
            "HPO license (free for research with attribution)",
            "phenotype nodes and hierarchy",
        ),
        Source(
            "hpo_g2p_T",
            "HPO ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype",
            "Jenkins monthly build captured 2019-09-02",
            "2019-09-02",
            "http://web.archive.org/web/20190902195916/http://compbio.charite.de:80/jenkins/job/hpo.annotations.monthly/lastSuccessfulBuild/artifact/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt",
            "HPO license",
            "gene-phenotype edges",
            note="Original Jenkins server is offline; Internet Archive capture. The disease-phenotype file has no complete pre-T capture (only truncated ones), so disease-phenotype edges are not included at T.",
        ),
        Source(
            "gwas_T",
            "NHGRI-EBI GWAS Catalog associations (ontology annotated)",
            "release 2019-12-16",
            "2019-12-16",
            "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/2019/12/16/gwas-catalog-associations_ontology-annotated.tsv",
            "CC0 (GWAS Catalog terms of use)",
            "variant-trait edges (additionally filtered to DATE ADDED TO CATALOG < T)",
        ),
        Source(
            "primekg",
            "PrimeKG",
            "Harvard Dataverse doi:10.7910/DVN/IXA7BM (kg.csv)",
            "2022-04-01",
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM",
            "CC BY 4.0 (some sources have other terms)",
            "LEAKY sensitivity background only; STaRK-Prime (M6)",
            leaky=True,
            note="Built from source releases up to 2021, so it postdates T. Not used in the primary temporal experiment.",
        ),
        Source(
            "gnomad_v2",
            "gnomAD",
            "v2.1.1",
            "2019-03-06",
            "https://gnomad.broadinstitute.org",
            "ODC Open Database License (data), CC0 summary statistics",
            "allele frequency features at T",
        ),
        Source(
            "gnomad_v4",
            "gnomAD",
            "v4.1",
            "2024-04-19",
            "https://gnomad.broadinstitute.org",
            "ODbL",
            "dossier 'latest' mode only; leaky for T",
            leaky=True,
        ),
    ]
}


def manifest() -> list[dict]:
    return [asdict(s) for s in SOURCES.values()]


def time_safe_keys(t_date: str = T_DATE) -> list[str]:
    return [k for k, s in SOURCES.items() if not s.leaky and s.release_date <= t_date]
