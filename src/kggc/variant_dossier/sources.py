"""Evidence fetchers for the variant dossier. Every call goes through the disk cache.

Each fetcher returns a *record*: ``{"record_id", "source", "source_version", "url",
"retrieved_at", "data"}`` where ``data`` is the raw (or minimally trimmed) API payload. Claims
in the dossier cite ``record_id`` plus a path into ``data``; the verifier re-reads the same
record.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from kggc.data.cache import CachedClient, default_client

ENSEMBL = "https://rest.ensembl.org"
DBSNP = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{}"
MYVARIANT = "https://myvariant.info/v1/query"
GNOMAD = "https://gnomad.broadinstitute.org/api"
GWAS_V2 = "https://www.ebi.ac.uk/gwas/rest/api/v2/associations"
LITVAR = (
    "https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/litvar%40{}%23%23/publications"
)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CIVIC = "https://civicdb.org/api/graphql"

GNOMAD_V2_POPS = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas", "oth"]
GNOMAD_V4_POPS = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas", "mid", "ami", "remaining"]
POP_NAMES = {
    "afr": "African/African American",
    "amr": "Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "nfe": "Non-Finnish European",
    "sas": "South Asian",
    "oth": "Other",
    "mid": "Middle Eastern",
    "ami": "Amish",
    "remaining": "Remaining",
}


def _record(rid: str, source: str, rec: dict, data: Any = None, version: str | None = None) -> dict:
    return {
        "record_id": rid,
        "source": source,
        "source_version": version or rec.get("source_version"),
        "url": rec["request"]["url"],
        "retrieved_at": rec["retrieved_at"],
        "status": rec["status"],
        "data": rec["body"] if data is None else data,
    }


class Sources:
    def __init__(self, client: CachedClient | None = None) -> None:
        self.c = client or default_client()

    # ---------------------------------------------------------------- identity
    def dbsnp(self, rsid: str) -> dict:
        num = rsid.lower().replace("rs", "")
        rec = self.c.request("ncbi", DBSNP.format(num), source_version="dbSNP (build in record)")
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        data: dict[str, Any] = {
            "refsnp_id": body.get("refsnp_id"),
            "last_update_build_id": body.get("last_update_build_id"),
            "dbsnp1_merges": [m.get("merged_rsid") for m in body.get("dbsnp1_merges", [])],
            "merged_into": (body.get("merged_snapshot_data") or {}).get("merged_into", []),
            "citations": body.get("citations", []),
        }
        ps = body.get("primary_snapshot_data")
        if ps:
            data["variant_type"] = ps.get("variant_type")
            ptlp = [p for p in ps.get("placements_with_allele", []) if p.get("is_ptlp")]
            if ptlp:
                p = ptlp[0]
                data["seq_id"] = p.get("seq_id")
                data["alleles"] = [
                    {
                        "hgvs": a.get("hgvs"),
                        "position0": a["allele"]["spdi"]["position"],
                        "deleted": a["allele"]["spdi"]["deleted_sequence"],
                        "inserted": a["allele"]["spdi"]["inserted_sequence"],
                    }
                    for a in p.get("alleles", [])
                    if "spdi" in a.get("allele", {})
                ]
            genes = []
            for ann in ps.get("allele_annotations", []):
                for asm in ann.get("assembly_annotation", []):
                    for g in asm.get("genes", []):
                        if g.get("locus") and g["locus"] not in genes:
                            genes.append(g["locus"])
            data["genes"] = genes
        build = body.get("last_update_build_id")
        return _record(f"dbSNP:{rsid}", "dbSNP", rec, data, f"dbSNP build {build}")

    def vep(self, rsid: str) -> dict:
        rec = self.c.request(
            "ensembl",
            f"{ENSEMBL}/vep/human/id/{rsid}",
            params={
                "content-type": "application/json",
                "hgvs": 1,
                "mane": 1,
                "pick_allele": 1,
                "variant_class": 1,
                "numbers": 1,
            },
            source_version="Ensembl VEP REST (GRCh38)",
        )
        body = rec["body"]
        data: dict[str, Any] = {}
        if isinstance(body, list) and body:
            b = body[0]
            data = {
                "id": b.get("id"),
                "assembly_name": b.get("assembly_name"),
                "seq_region_name": b.get("seq_region_name"),
                "start": b.get("start"),
                "end": b.get("end"),
                "allele_string": b.get("allele_string"),
                "variant_class": b.get("variant_class"),
                "most_severe_consequence": b.get("most_severe_consequence"),
                "transcript_consequences": [
                    {
                        k: tc.get(k)
                        for k in (
                            "variant_allele",
                            "gene_symbol",
                            "gene_id",
                            "transcript_id",
                            "mane_select",
                            "consequence_terms",
                            "hgvsc",
                            "hgvsp",
                            "impact",
                            "amino_acids",
                            "protein_start",
                            "biotype",
                        )
                    }
                    for tc in b.get("transcript_consequences", [])
                ],
                "regulatory_or_intergenic": [
                    {k: x.get(k) for k in ("variant_allele", "consequence_terms", "impact")}
                    for x in b.get("intergenic_consequences", [])
                    + b.get("regulatory_feature_consequences", [])
                ],
                "cosmic_ids": sorted(
                    {
                        c.get("id")
                        for c in b.get("colocated_variants", [])
                        if str(c.get("id", "")).startswith("COSV")
                    }
                ),
                "colocated_somatic": any(c.get("somatic") for c in b.get("colocated_variants", [])),
            }
        return _record(f"VEP:{rsid}", "Ensembl VEP", rec, data)

    # ---------------------------------------------------------------- frequency
    def myvariant(self, rsid: str) -> dict:
        rec = self.c.request(
            "myvariant",
            MYVARIANT,
            params={
                "q": f"dbsnp.rsid:{rsid}",
                "fields": "gnomad_exome.af,gnomad_exome.an,gnomad_genome.af,gnomad_genome.an,"
                "gnomad_exome.alt,gnomad_exome.ref,gnomad_genome.alt,gnomad_genome.ref,"
                "cosmic,docm.disease,docm.pubmed_id,civic.id,civic.name,civic.feature,"
                "civic.molecularProfiles,cadd.phred,dbsnp.vartype,dbsnp.ref,dbsnp.alt,vcf",
                "size": 20,
            },
            source_version="MyVariant.info (gnomAD v2.1.1, GRCh37)",
        )
        hits = rec["body"].get("hits", []) if isinstance(rec["body"], dict) else []
        return _record(
            f"gnomAD-v2.1.1:{rsid}", "gnomAD v2.1.1 (via MyVariant.info)", rec, {"hits": hits}
        )

    def gnomad_v4(self, variant_id: str) -> dict:
        query = (
            "query($v: String!) { variant(variantId: $v, dataset: gnomad_r4) { variant_id rsids "
            "exome { ac an populations { id ac an } } genome { ac an populations { id ac an } } } }"
        )
        rec = self.c.request(
            "gnomad",
            GNOMAD,
            method="POST",
            json_body={"query": query, "variables": {"v": variant_id}},
            source_version="gnomAD v4.1 (GRCh38)",
        )
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        var = (body.get("data") or {}).get("variant")
        data = {"variant": var, "errors": body.get("errors")}
        if var:
            for part in ("exome", "genome"):
                if var.get(part):
                    var[part]["populations"] = [
                        p for p in var[part]["populations"] if p["id"] in GNOMAD_V4_POPS
                    ]
        return _record(f"gnomAD-v4.1:{variant_id}", "gnomAD v4.1", rec, data)

    # ---------------------------------------------------------------- associations
    def gwas(self, rsid: str, max_pages: int = 10) -> dict:
        assoc: list[dict] = []
        first = None
        for page in range(max_pages):
            rec = self.c.request(
                "gwas",
                GWAS_V2,
                params={"rs_id": rsid, "size": 200, "page": page},
                source_version="NHGRI-EBI GWAS Catalog REST API v2",
            )
            first = first or rec
            body = rec["body"] if isinstance(rec["body"], dict) else {}
            items = (body.get("_embedded") or {}).get("associations", [])
            for a in items:
                assoc.append(
                    {
                        k: a.get(k)
                        for k in (
                            "association_id",
                            "accession_id",
                            "pubmed_id",
                            "first_author",
                            "p_value",
                            "pvalue_mantissa",
                            "pvalue_exponent",
                            "or_value",
                            "beta",
                            "risk_frequency",
                            "efo_traits",
                            "reported_trait",
                            "mapped_genes",
                            "snp_effect_allele",
                            "multi_snp_haplotype",
                        )
                    }
                )
            pg = body.get("page") or {}
            if not items or page + 1 >= int(pg.get("totalPages", 1)):
                break
        return _record(f"GWAS:{rsid}", "GWAS Catalog", first, {"associations": assoc})

    # ---------------------------------------------------------------- literature
    def litvar(self, rsid: str) -> dict:
        rec = self.c.request("litvar", LITVAR.format(rsid), source_version="LitVar2")
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        return _record(f"LitVar2:{rsid}", "LitVar2", rec, {"pmids": body.get("pmids", [])})

    def pubmed_summaries(self, pmids: list[int | str]) -> dict[str, dict]:
        """PMID -> {title, journal, pubdate, sortpubdate}; batched esummary calls."""
        out: dict[str, dict] = {}
        ids = sorted({str(p) for p in pmids if str(p).isdigit()}, key=int)
        for i in range(0, len(ids), 200):
            chunk = ids[i : i + 200]
            rec = self.c.request(
                "ncbi",
                f"{EUTILS}/esummary.fcgi",
                method="POST",
                data={"db": "pubmed", "id": ",".join(chunk), "retmode": "json"},
                source_version="PubMed E-utilities",
            )
            res = rec["body"].get("result", {}) if isinstance(rec["body"], dict) else {}
            for pid in chunk:
                r = res.get(pid)
                if not isinstance(r, dict) or "error" in r:
                    continue
                out[pid] = {
                    "pmid": pid,
                    "title": r.get("title", ""),
                    "journal": r.get("source", ""),
                    "pubdate": r.get("pubdate", ""),
                    "sortpubdate": (r.get("sortpubdate", "") or "")[:10].replace("/", "-"),
                    "retrieved_at": rec["retrieved_at"],
                }
        return out

    def pubmed_abstracts(self, pmids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        ids = [str(p) for p in pmids]
        if not ids:
            return out
        rec = self.c.request(
            "ncbi",
            f"{EUTILS}/efetch.fcgi",
            method="POST",
            data={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"},
            source_version="PubMed E-utilities",
            expect_json=False,
        )
        try:
            root = ET.fromstring(rec["body"])
        except (ET.ParseError, TypeError):
            return out
        for art in root.findall(".//PubmedArticle"):
            pid = art.findtext(".//PMID")
            parts = ["".join(x.itertext()) for x in art.findall(".//Abstract/AbstractText")]
            out[pid] = " ".join(p.strip() for p in parts if p)
        return out

    # ---------------------------------------------------------------- somatic
    def civic(self, gene: str, variant_name: str) -> dict:
        query = (
            "query($g: String, $v: String) { browseVariants(featureName: $g, variantName: $v, first: 5)"
            " { nodes { id name featureName link } } }"
        )
        rec = self.c.request(
            "civic",
            CIVIC,
            method="POST",
            json_body={"query": query, "variables": {"g": gene, "v": variant_name}},
            source_version="CIViC GraphQL API",
        )
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        nodes = ((body.get("data") or {}).get("browseVariants") or {}).get("nodes", [])
        nodes = [n for n in nodes if n.get("name", "").upper() == variant_name.upper()]
        detail = []
        for n in nodes[:1]:
            q2 = (
                "query($id: Int!) { variant(id: $id) { id name molecularProfiles(first: 20) { nodes "
                "{ id name evidenceItems(first: 1) { totalCount } } } } }"
            )
            r2 = self.c.request(
                "civic",
                CIVIC,
                method="POST",
                json_body={"query": q2, "variables": {"id": n["id"]}},
                source_version="CIViC GraphQL API",
            )
            v = (
                ((r2["body"] or {}).get("data") or {}).get("variant")
                if isinstance(r2["body"], dict)
                else None
            )
            if v:
                mps = (v.get("molecularProfiles") or {}).get("nodes", [])
                detail.append(
                    {
                        "id": v["id"],
                        "name": v["name"],
                        "gene": n.get("featureName"),
                        "n_molecular_profiles": len(mps),
                        "n_evidence_items": sum(
                            (m.get("evidenceItems") or {}).get("totalCount", 0) for m in mps
                        ),
                    }
                )
        return _record(f"CIViC:{gene}:{variant_name}", "CIViC", rec, {"variants": detail})


_AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*",
}  # fmt: skip


def short_protein_change(hgvsp: str | None) -> str | None:
    """'ENSP..:p.Gly12Asp' -> 'G12D' (simple substitutions only)."""
    if not hgvsp:
        return None
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", hgvsp)
    if not m or m.group(1) not in _AA3 or m.group(3) not in _AA3:
        return None
    return f"{_AA3[m.group(1)]}{m.group(2)}{_AA3[m.group(3)]}"
