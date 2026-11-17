"""Variant annotation from MyVariant.info (gnomAD v2.1.1 frequencies, CADD).

MyVariant.info's metadata endpoint reports gnomAD version 2.1.1 (released 2019-03-06, before T).
It reports no version for CADD, so CADD is kept out of the primary time-T feature set and only
used in a separately labeled analysis (see NOTES.md).

Variants are looked up by ClinVar VariationID (``scopes=clinvar.variant_id``). This uses
today's ClinVar only as an identifier crosswalk to genomic coordinates; no classification
field is requested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kggc.data.cache import CachedClient, default_client

MYVARIANT_URL = "https://myvariant.info/v1/query"
FIELDS = "gnomad_exome.af,gnomad_genome.af,cadd.phred,cadd.consequence"
POPS = ["afr", "amr", "asj", "eas", "fin", "nfe", "sas", "oth"]


def _af(doc: dict, key: str, field: str = "af") -> float:
    v = doc.get(key, {})
    if isinstance(v, list):
        v = v[0] if v else {}
    af = v.get("af", {}) if isinstance(v, dict) else {}
    val = af.get(field) if isinstance(af, dict) else None
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _cadd(doc: dict) -> float:
    c = doc.get("cadd", {})
    if isinstance(c, list):
        c = c[0] if c else {}
    v = c.get("phred") if isinstance(c, dict) else None
    if isinstance(v, list):
        v = max(v) if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def parse_hits(hits: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for h in hits:
        q = str(h.get("query"))
        if h.get("notfound"):
            best.setdefault(q, {"variation_id": int(q), "found": False})
            continue
        if q in best and best[q].get("found"):
            continue  # keep the top-scoring hit only
        rec = {
            "variation_id": int(q),
            "found": True,
            "hgvs_hg19": h.get("_id"),
            "gnomad_exome_af": _af(h, "gnomad_exome"),
            "gnomad_genome_af": _af(h, "gnomad_genome"),
            "cadd_phred": _cadd(h),
        }
        for pop in POPS:
            rec[f"gnomad_exome_af_{pop}"] = _af(h, "gnomad_exome", f"af_{pop}")
        best[q] = rec
    return list(best.values())


def fetch_annotations(
    variation_ids: list[int], client: CachedClient | None = None, batch: int = 1000
) -> pd.DataFrame:
    client = client or default_client()
    ids = sorted({int(v) for v in variation_ids})
    rows: list[dict] = []
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        body = client.get_json(
            "myvariant",
            MYVARIANT_URL,
            method="POST",
            data={
                "q": ",".join(map(str, chunk)),
                "scopes": "clinvar.variant_id",
                "fields": FIELDS,
                "assembly": "hg19",
            },
            source_version="gnomAD v2.1.1 via MyVariant.info; CADD version not reported",
        )
        if not isinstance(body, list):
            raise RuntimeError(f"unexpected MyVariant response: {str(body)[:200]}")
        rows += parse_hits(body)
    df = pd.DataFrame(rows)
    # A variant absent from gnomAD v2.1.1 was not observed there: AF is treated as 0 with a flag.
    df["gnomad_af"] = df[["gnomad_exome_af", "gnomad_genome_af"]].max(axis=1)
    df["in_gnomad"] = df["gnomad_af"].notna().astype(int)
    df["gnomad_af"] = df["gnomad_af"].fillna(0.0)
    return df
