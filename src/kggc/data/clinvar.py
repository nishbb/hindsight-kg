"""ClinVar ``variant_summary.txt.gz`` loader that works across 2018-2026 file formats.

Format differences handled here (verified on the 2020-01 and 2026-11 files):
* Conditions are separated by ``;`` in older releases and by ``|`` in newer ones.
  Within a condition, cross-references are separated by ``,``.
* Conflicting assertions are "Conflicting interpretations of pathogenicity" in older releases
  and "Conflicting classifications of pathogenicity" in newer ones.
* Newer files add VCF-style position columns and somatic/oncogenicity columns; we only use
  columns that exist in every release.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

USECOLS = [
    "#AlleleID",
    "Type",
    "Name",
    "GeneID",
    "GeneSymbol",
    "ClinicalSignificance",
    "LastEvaluated",
    "RS# (dbSNP)",
    "RCVaccession",
    "PhenotypeIDS",
    "PhenotypeList",
    "OriginSimple",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "ReviewStatus",
    "NumberSubmitters",
    "VariationID",
]

# Exact-match mapping only. Anything else (compound terms such as "Pathogenic, risk factor",
# "low penetrance" subclasses, drug response, etc.) is OTHER and excluded from labels.
CLNSIG_MAP = {
    "Pathogenic": "PLP",
    "Likely pathogenic": "PLP",
    "Pathogenic/Likely pathogenic": "PLP",
    "Uncertain significance": "VUS",
    "Benign": "BLB",
    "Likely benign": "BLB",
    "Benign/Likely benign": "BLB",
    "Conflicting interpretations of pathogenicity": "CONFLICT",
    "Conflicting classifications of pathogenicity": "CONFLICT",
}

REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, conflicting classifications": 1,
}

NON_CONDITIONS = {"not provided", "not specified", "see cases", "-", ""}


def snapshot_date_from_name(path: str | Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", Path(path).name)
    return m.group(1) if m else "latest"


def split_conditions(ids_field: str, names_field: str) -> list[tuple[str, list[str]]]:
    """Return [(condition_name, [xref, ...]), ...] for one row, in RCV order."""
    ids_field = ids_field if isinstance(ids_field, str) else ""
    names_field = names_field if isinstance(names_field, str) else ""
    sep = "|" if ("|" in ids_field or "|" in names_field) else ";"
    ids = ids_field.split(sep) if ids_field else []
    names = names_field.split(sep) if names_field else []
    n = max(len(ids), len(names))
    ids += [""] * (n - len(ids))
    names += [""] * (n - len(names))
    out = []
    for nm, xr in zip(names, ids):
        xrefs = [x for x in xr.split(",") if x and x != "-"]
        out.append((nm.strip(), xrefs))
    return out


def informative_conditions(conds: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    keep = []
    for nm, xr in conds:
        if nm.strip().lower() in NON_CONDITIONS:
            continue
        keep.append((nm, xr))
    return keep


def classify(sig: str) -> str:
    return CLNSIG_MAP.get(str(sig).strip(), "OTHER")


def load_variant_summary(
    path: str | Path,
    assembly: str = "GRCh38",
    allele_to_variation: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per VariationID (preferring the requested assembly).

    Releases before mid-2018 have no VariationID column; pass ``allele_to_variation``
    (AlleleID -> VariationID, taken from a later release) to recover it.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=lambda c: c in USECOLS,
        dtype=str,
        low_memory=False,
        na_filter=False,
    )
    df = df.rename(
        columns={
            "#AlleleID": "allele_id",
            "Type": "type",
            "Name": "name",
            "GeneID": "gene_id",
            "GeneSymbol": "gene_symbol",
            "ClinicalSignificance": "clnsig",
            "LastEvaluated": "last_evaluated",
            "RS# (dbSNP)": "rsid",
            "RCVaccession": "rcv",
            "PhenotypeIDS": "phenotype_ids",
            "PhenotypeList": "phenotype_list",
            "OriginSimple": "origin_simple",
            "Assembly": "assembly",
            "Chromosome": "chrom",
            "Start": "start",
            "Stop": "stop",
            "ReferenceAllele": "ref",
            "AlternateAllele": "alt",
            "ReviewStatus": "review_status",
            "NumberSubmitters": "n_submitters",
            "VariationID": "variation_id",
        }
    )
    if "variation_id" not in df.columns:
        if allele_to_variation is None:
            raise ValueError(f"{path} has no VariationID column; pass allele_to_variation")
        df["variation_id"] = df["allele_id"].map(allele_to_variation)
    df["variation_id"] = pd.to_numeric(df["variation_id"], errors="coerce").astype("Int64")
    df = df[df["variation_id"].notna()].copy()
    df["_pref"] = (df["assembly"] != assembly).astype(int)
    df = df.sort_values(["variation_id", "_pref"], kind="stable")
    df = df.drop_duplicates("variation_id", keep="first").drop(columns="_pref")
    df["clnsig_class"] = df["clnsig"].map(classify)
    df["stars"] = df["review_status"].map(REVIEW_STARS).fillna(0).astype(int)
    df["n_submitters"] = pd.to_numeric(df["n_submitters"], errors="coerce").fillna(0).astype(int)
    df["gene_id"] = pd.to_numeric(df["gene_id"], errors="coerce").fillna(-1).astype(int)
    df["rsid"] = np.where(
        df["rsid"].str.fullmatch(r"\d+") & (df["rsid"] != "-1"), "rs" + df["rsid"], ""
    )
    df["snapshot"] = snapshot_date_from_name(path)
    return df.reset_index(drop=True)


_PROT_RE = re.compile(r"\(p\.([^)]*)\)")


def consequence_from_name(name: str, vtype: str) -> str:
    """Coarse molecular consequence from the HGVS name. Uses only the variant itself."""
    name = name or ""
    m = _PROT_RE.search(name)
    if m:
        p = m.group(1)
        if p.endswith("=") or p == "=":
            return "synonymous"
        if "fs" in p:
            return "frameshift"
        if "Ter" in p or p.endswith("*"):
            return "nonsense"
        if "delins" in p or "ins" in p or "del" in p or "dup" in p:
            return "inframe_indel"
        if p.startswith("Met1"):
            return "start_lost"
        return "missense"
    if re.search(r"c\.[-*]?\d+[+-][12][ACGT_]", name) or re.search(r"c\.\d+[+-][12]del", name):
        return "splice_site"
    if re.search(r"c\.[-*]?\d+[+-]\d+", name):
        return "intronic"
    if "c.-" in name:
        return "utr5"
    if "c.*" in name:
        return "utr3"
    if vtype and vtype.lower() in ("copy number loss", "copy number gain", "deletion"):
        return "structural_or_other"
    return "other"
