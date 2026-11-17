"""Download every raw input at a pinned version and verify its MD5.

Usage: python scripts/download_data.py [--minimal]
--minimal skips files only needed for the dossier's assertion history (submission summaries).
Files go to data/raw/<source>/. Existing files with the right checksum are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

from kggc.utils import RAW_DIR

CV = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive"
FILES = [
    # (relative path, url, md5, needed for --minimal)
    (
        "clinvar/variant_summary_2018-01.txt.gz",
        f"{CV}/2018/variant_summary_2018-01.txt.gz",
        "d1d35961a797dde4eacc27ef869edb72",
        True,
    ),
    (
        "clinvar/variant_summary_2019-01.txt.gz",
        f"{CV}/2019/variant_summary_2019-01.txt.gz",
        "0435e5ea130dcd0eff9b241bca6b56b8",
        True,
    ),
    (
        "clinvar/variant_summary_2020-01.txt.gz",
        f"{CV}/2020/variant_summary_2020-01.txt.gz",
        "77cc8d4f887beb45cc5a28624bfbb8b0",
        True,
    ),
    (
        "clinvar/variant_summary_2026-11.txt.gz",
        f"{CV}/variant_summary_2026-11.txt.gz",
        "d7e0d7e581e220100fc84ca64fe953ba",
        True,
    ),
    (
        "clinvar/submission_summary_2020-01.txt.gz",
        f"{CV}/2020/submission_summary_2020-01.txt.gz",
        "71140069399c938271d61be67cd7760c",
        False,
    ),
    (
        "clinvar/submission_summary_2026-11.txt.gz",
        f"{CV}/submission_summary_2026-11.txt.gz",
        "56a4d843505e766e52022edfa37f8910",
        False,
    ),
    (
        "hetionet/hetionet-v1.0-edges.sif.gz",
        "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
        "b4b2ab8890d403c2077a65e0f34fc4d9",
        True,
    ),
    (
        "hetionet/hetionet-v1.0-nodes.tsv",
        "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv",
        "d2789861ab5b977dcee3c23ac9c9f7ce",
        True,
    ),
    (
        "mondo/mondo_2019-11-25.obo",
        "https://github.com/monarch-initiative/mondo/releases/download/v2019-11-25/mondo.obo",
        "265399624a6660b7b2500ea94220b68c",
        True,
    ),
    (
        "hpo/hp_2019-11-08.obo",
        "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/v2019-11-08/hp.obo",
        "53c9c324bb9e5da0891d78d09c27a4fb",
        True,
    ),
    (
        "hpo/genes_to_phenotype_2019-09-02.txt",
        "http://web.archive.org/web/20190902195916id_/http://compbio.charite.de:80/jenkins/job/hpo.annotations.monthly/lastSuccessfulBuild/artifact/annotation/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt",
        "04e5bcda70c64464342567352e3cd30f",
        True,
    ),
    (
        "gwas/gwas-catalog-associations_ontology-annotated_2019-12-16.tsv",
        "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/2019/12/16/gwas-catalog-associations_ontology-annotated.tsv",
        "bb3e69fe1e8297667c81eae18d9e7200",
        True,
    ),
    # HGNC is a living file used only for gene-family stratification of reports; checksum not pinned
    (
        "hgnc/hgnc_complete_set.txt",
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        None,
        True,
    ),
]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true")
    args = ap.parse_args()
    bad = []
    for rel, url, want, minimal in FILES:
        if args.minimal and not minimal:
            continue
        dest = RAW_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (want is None or md5(dest) == want):
            print(f"ok      {rel}")
            continue
        print(f"fetch   {rel}", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        got = md5(tmp)
        if want is not None and got != want:
            bad.append(f"{rel}: md5 {got} != {want}")
            print(f"BAD     {rel}", file=sys.stderr)
            continue
        tmp.replace(dest)
    if bad:
        sys.exit("checksum failures:\n" + "\n".join(bad))


if __name__ == "__main__":
    main()
