"""Local, dated ClinVar index (SQLite) built from archived tab-delimited releases.

Using archived releases instead of the live API makes ``--as-of`` queries exact: a dossier
as of 2020-01-01 sees only the releases published on or before that date, and the assertion
history is the sequence of classifications across releases.
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import pandas as pd

from kggc.utils import PROC_DIR, RAW_DIR

# snapshot tag -> (variant_summary file, release date, submission_summary file or None)
SNAPSHOTS = {
    "2018-01": ("variant_summary_2018-01.txt.gz", "2018-01-04", None),
    "2019-01": ("variant_summary_2019-01.txt.gz", "2019-01-03", None),
    "2020-01": (
        "variant_summary_2020-01.txt.gz",
        "2020-01-02",
        "submission_summary_2020-01.txt.gz",
    ),
    "2026-11": ("variant_summary.txt.gz", "2026-11-16", "submission_summary.txt.gz"),
}
LATEST = "2026-11"
DB_PATH = PROC_DIR / "clinvar_local.sqlite"

VAR_COLS = {
    "#AlleleID": "allele_id",
    "Type": "type",
    "Name": "name",
    "GeneSymbol": "gene_symbol",
    "ClinicalSignificance": "clnsig",
    "LastEvaluated": "last_evaluated",
    "RS# (dbSNP)": "rs",
    "RCVaccession": "rcv",
    "PhenotypeList": "phenotype_list",
    "Origin": "origin",
    "OriginSimple": "origin_simple",
    "Assembly": "assembly",
    "Chromosome": "chrom",
    "Start": "start",
    "ReferenceAllele": "ref",
    "AlternateAllele": "alt",
    "ReviewStatus": "review_status",
    "NumberSubmitters": "n_submitters",
    "VariationID": "variation_id",
    "ReferenceAlleleVCF": "ref_vcf",
    "AlternateAlleleVCF": "alt_vcf",
    "PositionVCF": "pos_vcf",
    "SomaticClinicalImpact": "somatic_clinical_impact",
    "Oncogenicity": "oncogenicity",
}
SUB_COLS = {
    "#VariationID": "variation_id",
    "ClinicalSignificance": "clnsig",
    "DateLastEvaluated": "date_last_evaluated",
    "ReportedPhenotypeInfo": "reported_phenotype",
    "ReviewStatus": "review_status",
    "CollectionMethod": "collection_method",
    "OriginCounts": "origin_counts",
    "Submitter": "submitter",
    "SCV": "scv",
}


def _sub_header(path: Path) -> tuple[list[str], int]:
    """submission_summary has '#' comment lines; the last one is the column header."""
    n, header = 0, None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                header = line.rstrip("\n").split("\t")
                n += 1
            else:
                break
    return header, n


def build_index(raw_dir: Path | None = None, db_path: Path | None = None, log=print) -> Path:
    raw = Path(raw_dir or RAW_DIR / "clinvar")
    db_path = Path(db_path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE meta (snapshot TEXT, release_date TEXT, has_submissions INT)")
    for tag, (vfile, date, sfile) in SNAPSHOTS.items():
        vp = raw / vfile
        if not vp.exists():
            log(f"[clinvar index] missing {vp}, skipping snapshot {tag}")
            continue
        for chunk in pd.read_csv(
            vp,
            sep="\t",
            dtype=str,
            na_filter=False,
            chunksize=500_000,
            usecols=lambda c: c in VAR_COLS,
        ):
            chunk = chunk.rename(columns=VAR_COLS)
            chunk = chunk[(chunk["rs"] != "-1") & (chunk["rs"] != "") & (chunk["rs"] != "-")]
            chunk = chunk[chunk["assembly"].isin(["GRCh38", "GRCh37"])]
            for c in VAR_COLS.values():
                if c not in chunk:
                    chunk[c] = ""
            chunk["rsid"] = "rs" + chunk["rs"]
            chunk["snapshot"] = tag
            chunk.drop(columns=["rs"]).to_sql("variants", con, if_exists="append", index=False)
        has_sub = 0
        if sfile and (raw / sfile).exists():
            header, nskip = _sub_header(raw / sfile)
            for chunk in pd.read_csv(
                raw / sfile,
                sep="\t",
                dtype=str,
                na_filter=False,
                chunksize=500_000,
                skiprows=nskip,
                names=header,
                usecols=lambda c: c in SUB_COLS,
                quoting=3,
                on_bad_lines="skip",
            ):
                chunk = chunk.rename(columns=SUB_COLS)
                chunk["snapshot"] = tag
                chunk.to_sql("submissions", con, if_exists="append", index=False)
            has_sub = 1
        con.execute("INSERT INTO meta VALUES (?,?,?)", (tag, date, has_sub))
        con.commit()
        log(f"[clinvar index] indexed snapshot {tag}")
    con.execute("CREATE INDEX ix_var_rs ON variants (rsid, snapshot)")
    con.execute("CREATE INDEX ix_var_allele ON variants (allele_id, snapshot)")
    try:
        con.execute("CREATE INDEX ix_sub ON submissions (variation_id, snapshot)")
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()
    tmp.replace(db_path)
    return db_path


class ClinVarLocal:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or DB_PATH)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"{self.db_path} not found. Run `python scripts/build_clinvar_index.py` first."
            )
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self.meta = {
            r["snapshot"]: dict(r) for r in self.con.execute("SELECT * FROM meta").fetchall()
        }

    def snapshots_as_of(self, as_of: str) -> list[str]:
        """Snapshot tags released on or before ``as_of`` (ISO date or 'latest'), oldest first."""
        tags = sorted(self.meta, key=lambda t: self.meta[t]["release_date"])
        if as_of == "latest":
            return tags
        return [t for t in tags if self.meta[t]["release_date"] <= as_of]

    def variants(self, rsid: str, snapshot: str) -> list[dict]:
        rows = self.con.execute(
            "SELECT * FROM variants WHERE rsid=? AND snapshot=?", (rsid, snapshot)
        ).fetchall()
        best: dict[str, dict] = {}
        for r in map(dict, rows):
            k = r["allele_id"]
            if k not in best or r["assembly"] == "GRCh38":
                best[k] = r
        return list(best.values())

    def history(self, allele_id: str, snapshots: list[str]) -> list[dict]:
        out = []
        for s in snapshots:
            r = self.con.execute(
                "SELECT snapshot, clnsig, review_status, n_submitters, last_evaluated FROM variants "
                "WHERE allele_id=? AND snapshot=? LIMIT 1",
                (allele_id, s),
            ).fetchone()
            if r:
                d = dict(r)
                d["release_date"] = self.meta[s]["release_date"]
                out.append(d)
        return out

    def submissions(self, variation_id: str, snapshot: str) -> list[dict]:
        if not self.meta.get(snapshot, {}).get("has_submissions"):
            return []
        rows = self.con.execute(
            "SELECT * FROM submissions WHERE variation_id=? AND snapshot=?",
            (str(variation_id), snapshot),
        ).fetchall()
        return [dict(r) for r in rows]
