"""Assemble a per-variant dossier: gather evidence, reconcile, and emit cited claims.

All sentences are generated from templates filled with values read from stored raw records.
No language model writes dossier text, so a sentence cannot contain a fact that is absent from
its cited record; ``kggc.variant_dossier.verify`` re-checks this mechanically.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from kggc.data.clinvar import CLNSIG_MAP
from kggc.variant_dossier.claims import Citation, Claim
from kggc.variant_dossier.clinvar_local import LATEST, ClinVarLocal
from kggc.variant_dossier.sources import (
    GNOMAD_V2_POPS,
    GNOMAD_V4_POPS,
    POP_NAMES,
    Sources,
    short_protein_change,
)

GNOMAD_V2_RELEASE = "2019-03-06"
BA1_THRESHOLD = 0.05  # ACMG/AMP stand-alone benign frequency threshold
NON_CONDITIONS = {"not provided", "not specified", "see cases", "-", ""}
SOMATIC_SOURCES_NOTE = (
    "CIViC, COSMIC and DoCM entries carry no reliable public release date, so they are only "
    "used when as_of is 'latest'."
)


def fmt_af(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.3g}"


def fmt_p(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")


def assoc_p(a: dict) -> tuple[float, str]:
    """(sortable log10 p, display string) using mantissa/exponent so tiny p-values do not underflow."""
    m, e = a.get("pvalue_mantissa"), a.get("pvalue_exponent")
    if m is not None and e is not None:
        import math

        return math.log10(max(float(m), 1e-300)) + float(e), f"{m}e{e}"
    import math

    p = a.get("p_value")
    return (math.log10(p) if p else -400.0), fmt_p(p or 0.0)


def spdi_to_vcf(pos0: int, deleted: str, inserted: str) -> tuple[int, str, str]:
    ref, alt = deleted, inserted
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    pos = pos0 + 1
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return pos, ref, alt


def _indel_core(ref: str, alt: str) -> tuple[int, str]:
    """(length change, inserted or deleted sequence) after trimming shared prefix and suffix."""
    while ref and alt and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
    while ref and alt and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    return len(alt) - len(ref), (alt or ref)


def same_allele(r1: str, a1: str, r2: str, a2: str) -> bool:
    """Equal alleles, allowing for different left/right alignment of an indel in a repeat."""
    if (r1, a1) == (r2, a2):
        return True
    if len(r1) == len(a1) or len(r2) == len(a2):
        return False
    (d1, s1), (d2, s2) = _indel_core(r1, a1), _indel_core(r2, a2)
    return d1 == d2 and len(s1) == len(s2) and s1 in (s2 + s2)


def allele_type(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "single-nucleotide variant (SNV)"
    if len(ref) == len(alt):
        return "multi-nucleotide variant (MNV)"
    if len(ref) > len(alt):
        return "deletion" if alt and ref.startswith(alt) else "deletion-insertion (delins)"
    return (
        "insertion or duplication" if ref and alt.startswith(ref) else "deletion-insertion (delins)"
    )


def chrom_from_seq_id(seq_id: str) -> str | None:
    m = re.match(r"NC_0000(\d\d)\.", seq_id or "")
    if not m:
        return None
    n = int(m.group(1))
    return {23: "X", 24: "Y"}.get(n, str(n))


def split_conditions(lst: str) -> list[str]:
    parts = re.split(r"[|;]", lst or "")
    out = []
    for p in parts:
        p = p.strip()
        if p.lower() not in NON_CONDITIONS and p not in out:
            out.append(p)
    return out


def _norm_class(sig: str) -> str:
    return CLNSIG_MAP.get((sig or "").strip(), "OTHER")


class DossierBuilder:
    def __init__(self, sources: Sources | None = None, clinvar: ClinVarLocal | None = None) -> None:
        self.s = sources or Sources()
        self.cv = clinvar or ClinVarLocal()

    # ------------------------------------------------------------------ main
    def build(self, rsid: str, as_of: str = "latest", max_abstracts: int = 3) -> dict:
        rsid = rsid.strip().lower()
        latest_mode = as_of == "latest"
        cutoff = "9999-12-31" if latest_mode else as_of
        records: dict[str, dict] = {}
        claims: list[Claim] = []
        notes: list[str] = []

        def add(rec: dict) -> dict:
            records[rec["record_id"]] = rec
            return rec

        def claim(section: str, text: str, cits: list[Citation]) -> None:
            claims.append(Claim(section, text, cits))

        # ---------------- identity (dbSNP merge history, alleles)
        db_q = add(self.s.dbsnp(rsid))
        current = rsid
        if db_q["data"].get("merged_into"):
            current = "rs" + str(db_q["data"]["merged_into"][0])
            claim(
                "Identity",
                f"{rsid} is a retired dbSNP identifier that was merged into {current}; all evidence below is for {current}.",
                [
                    Citation(
                        db_q["record_id"],
                        "merged_into.0",
                        current[2:],
                        f"dbSNP {rsid}",
                        shown=current,
                    )
                ],
            )
        db = db_q if current == rsid else add(self.s.dbsnp(current))
        dbl = f"dbSNP {db['source_version'].replace('dbSNP ', '')} {current}"
        dd = db["data"]
        if current == rsid:
            claim(
                "Identity",
                f"{current} is a current dbSNP identifier (not merged into another record).",
                [Citation(db["record_id"], "merged_into", [], dbl)],
            )
        if dd.get("dbsnp1_merges"):
            old = ["rs" + str(x) for x in dd["dbsnp1_merges"]]
            shown = ", ".join(old[:6]) + (f" and {len(old) - 6} more" if len(old) > 6 else "")
            claim(
                "Identity",
                f"Earlier identifiers merged into {current}: {shown}.",
                [
                    Citation(db["record_id"], f"dbsnp1_merges.{i}", old[i][2:], dbl, shown=old[i])
                    for i in range(min(6, len(old)))
                ],
            )

        alleles: list[dict] = []
        chrom = chrom_from_seq_id(dd.get("seq_id", ""))
        for i, a in enumerate(dd.get("alleles", [])):
            if a["deleted"] == a["inserted"]:
                continue
            pos, ref, alt = spdi_to_vcf(a["position0"], a["deleted"], a["inserted"])
            alleles.append(
                {
                    "index": len(alleles) + 1,
                    "dbsnp_allele_index": i,
                    "hgvs_g": a["hgvs"],
                    "spdi_deleted": a["deleted"],
                    "spdi_inserted": a["inserted"],
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "type": allele_type(ref, alt),
                    "gnomad_v4_id": f"{chrom}-{pos}-{ref}-{alt}" if chrom and ref and alt else None,
                }
            )
        if dd.get("variant_type"):
            n = len(alleles)
            multi = (
                " It is multi-allelic, so each alternate allele is reported separately."
                if n > 1
                else ""
            )
            claim(
                "Identity",
                f"dbSNP records {current} as variant type '{dd['variant_type']}' on {dd.get('seq_id')} (GRCh38) with {n} alternate allele{'s' if n != 1 else ''}.{multi}",
                [
                    Citation(
                        db["record_id"],
                        "variant_type",
                        dd["variant_type"],
                        dbl,
                        shown=dd["variant_type"],
                    ),
                    Citation(
                        db["record_id"],
                        "seq_id",
                        dd.get("seq_id"),
                        dbl,
                        shown=str(dd.get("seq_id")),
                    ),
                ],
            )
        for a in alleles:
            claim(
                "Identity",
                f"Alternate allele {a['index']}: {a['hgvs_g']}, a {a['type']} ({a['ref']}>{a['alt']} in VCF form).",
                [
                    Citation(
                        db["record_id"],
                        f"alleles.{a['dbsnp_allele_index']}.hgvs",
                        a["hgvs_g"],
                        dbl,
                        shown=a["hgvs_g"],
                    )
                ],
            )
        if not alleles:
            notes.append("dbSNP returned no allele placements; allele-level sections are limited.")

        # ---------------- function (VEP)
        vep = add(self.s.vep(current))
        vl = f"Ensembl VEP {current}"
        vd = vep["data"]
        genes_vep: list[str] = []
        for a in alleles:
            a["vep"] = None
        for j, tc in enumerate(vd.get("transcript_consequences", []) or []):
            va = tc.get("variant_allele")
            target = next(
                (
                    a
                    for a in alleles
                    if va in (a["spdi_inserted"], a["alt"])
                    or (va == "-" and a["alt"] == a["ref"][:1])
                ),
                None,
            )
            if target is None and len(alleles) == 1:
                target = alleles[0]
            if tc.get("gene_symbol") and tc["gene_symbol"] not in genes_vep:
                genes_vep.append(tc["gene_symbol"])
            terms = ", ".join(tc.get("consequence_terms") or [])
            bits = [b for b in (tc.get("hgvsc"), tc.get("hgvsp")) if b]
            cits = [
                Citation(
                    vep["record_id"],
                    f"transcript_consequences.{j}.consequence_terms.0",
                    (tc.get("consequence_terms") or [None])[0],
                    vl,
                    shown=(tc.get("consequence_terms") or [""])[0],
                ),
                Citation(
                    vep["record_id"],
                    f"transcript_consequences.{j}.gene_symbol",
                    tc.get("gene_symbol"),
                    vl,
                    shown=tc.get("gene_symbol") or "",
                ),
            ]
            for key in ("hgvsc", "hgvsp"):
                if tc.get(key):
                    cits.append(
                        Citation(
                            vep["record_id"],
                            f"transcript_consequences.{j}.{key}",
                            tc[key],
                            vl,
                            shown=tc[key],
                        )
                    )
            label = (
                f"alternate allele {target['index']} ({target['ref']}>{target['alt']})"
                if target
                else f"allele '{va}'"
            )
            hg = f" ({'; '.join(bits)})" if bits else ""
            claim(
                "Function",
                f"For {label}, Ensembl VEP predicts {terms} in {tc.get('gene_symbol') or 'a transcript without a gene symbol'}{hg}, impact {tc.get('impact')}.",
                cits
                + [
                    Citation(
                        vep["record_id"],
                        f"transcript_consequences.{j}.impact",
                        tc.get("impact"),
                        vl,
                        shown=str(tc.get("impact")),
                    )
                ],
            )
            if target is not None:
                target["vep"] = {
                    k: tc.get(k)
                    for k in (
                        "gene_symbol",
                        "consequence_terms",
                        "hgvsc",
                        "hgvsp",
                        "impact",
                        "mane_select",
                    )
                }
        if not vd.get("transcript_consequences") and vd.get("most_severe_consequence"):
            claim(
                "Function",
                f"Ensembl VEP reports no transcript consequence; the most severe consequence is {vd['most_severe_consequence']}.",
                [
                    Citation(
                        vep["record_id"],
                        "most_severe_consequence",
                        vd["most_severe_consequence"],
                        vl,
                        shown=vd["most_severe_consequence"],
                    )
                ],
            )
        if vd.get("variant_class"):
            claim(
                "Function",
                f"Ensembl classifies the variant as '{vd['variant_class']}' with allele string {vd.get('allele_string')}.",
                [
                    Citation(
                        vep["record_id"],
                        "variant_class",
                        vd["variant_class"],
                        vl,
                        shown=vd["variant_class"],
                    ),
                    Citation(
                        vep["record_id"],
                        "allele_string",
                        vd.get("allele_string"),
                        vl,
                        shown=str(vd.get("allele_string")),
                    ),
                ],
            )
        if not vd:
            notes.append("Ensembl VEP returned no annotation for this identifier.")

        # ---------------- frequency
        mv = add(self.s.myvariant(current))
        hits = mv["data"].get("hits", [])
        use_v2 = cutoff >= GNOMAD_V2_RELEASE
        for a in alleles:
            a["frequency"] = {}
            if use_v2:
                self._freq_v2(a, hits, mv, claim, current)
            if latest_mode and a.get("gnomad_v4_id"):
                try:
                    g4 = add(self.s.gnomad_v4(a["gnomad_v4_id"]))
                    self._freq_v4(a, g4, claim)
                except Exception as err:  # network or API failure is reported, not hidden
                    notes.append(f"gnomAD v4.1 lookup failed for {a['gnomad_v4_id']}: {err}")
        if not use_v2:
            notes.append(
                f"gnomAD v2.1.1 was released {GNOMAD_V2_RELEASE}, after as_of={as_of}; no frequency reported."
            )

        # ---------------- ClinVar (germline), dated
        snaps = self.cv.snapshots_as_of(as_of)
        cur_snap = snaps[-1] if snaps else None
        clin_rows = self.cv.variants(current, LATEST)  # identifier crosswalk only
        if current != rsid:
            clin_rows += [r for r in self.cv.variants(rsid, LATEST)]
        clinvar_alleles: list[dict] = []
        if cur_snap:
            cur_rows = {r["allele_id"]: r for r in self.cv.variants(current, cur_snap)}
            crosswalk = {r["allele_id"]: r for r in clin_rows}
            for aid, row in cur_rows.items():
                xw = crosswalk.get(aid, row)
                vid = xw.get("variation_id") or row.get("variation_id") or ""
                rid = (
                    f"ClinVar:VCV{int(vid):09d}@{cur_snap}"
                    if str(vid).isdigit()
                    else f"ClinVar:allele{aid}@{cur_snap}"
                )
                subs = self.cv.submissions(vid, cur_snap) if str(vid).isdigit() else []
                hist = self.cv.history(aid, snaps)
                rec = add(
                    {
                        "record_id": rid,
                        "source": "ClinVar",
                        "source_version": f"ClinVar release {cur_snap} ({self.cv.meta[cur_snap]['release_date']})",
                        "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{vid}/",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "status": 200,
                        "data": {
                            **{k: row[k] for k in row if k not in ("snapshot",)},
                            "submissions": subs,
                            "history": hist,
                        },
                    }
                )
                target = None
                for a in alleles:
                    if xw.get("alt_vcf") and same_allele(
                        xw.get("ref_vcf"), xw.get("alt_vcf"), a["ref"], a["alt"]
                    ):
                        target = a
                if target is None and len(alleles) == 1 and len(cur_rows) == 1:
                    target = alleles[0]
                clinvar_alleles.append(
                    {
                        "row": row,
                        "record": rec,
                        "allele": target,
                        "history": hist,
                        "subs": subs,
                        "vid": vid,
                    }
                )
            self._clinvar_claims(clinvar_alleles, cur_snap, claim, current, latest_mode)
            if not cur_rows:
                rid = f"ClinVar:{current}@{cur_snap}"
                add(
                    {
                        "record_id": rid,
                        "source": "ClinVar",
                        "source_version": f"ClinVar release {cur_snap}",
                        "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "status": 200,
                        "data": {"rsid": current, "n_records": 0},
                    }
                )
                claim(
                    "Disease relevance (germline)",
                    f"The ClinVar {cur_snap} release contains no record for {current}.",
                    [Citation(rid, "n_records", 0, f"ClinVar {cur_snap} release")],
                )
        else:
            notes.append(f"No archived ClinVar release on or before as_of={as_of} is indexed.")

        # ---------------- somatic evidence (latest only)
        somatic = {"is_somatic_hotspot": False, "signals": []}
        if latest_mode:
            self._somatic_claims(
                alleles, clinvar_alleles, vep, mv, add, claim, somatic, current, notes
            )
        else:
            notes.append(SOMATIC_SOURCES_NOTE)

        # ---------------- GWAS
        gw = add(self.s.gwas(current))
        assoc_all = gw["data"]["associations"]
        pm = self.s.pubmed_summaries([a["pubmed_id"] for a in assoc_all if a.get("pubmed_id")])
        for i, a in enumerate(assoc_all):
            a["_index"] = i
            a["_pubdate"] = (pm.get(str(a.get("pubmed_id"))) or {}).get("sortpubdate", "")
        assoc = (
            [a for a in assoc_all if a["_pubdate"] and a["_pubdate"] <= cutoff]
            if not latest_mode
            else assoc_all
        )
        gwas_summary = self._gwas_claims(
            assoc, assoc_all, gw, pm, add, claim, current, as_of, latest_mode
        )

        # ---------------- literature
        lit = add(self.s.litvar(current))
        pmids = [str(p) for p in lit["data"].get("pmids", [])]
        summaries = self.s.pubmed_summaries(pmids)
        dated = [s for s in summaries.values() if s["sortpubdate"]]
        in_scope = sorted(
            [s for s in dated if latest_mode or s["sortpubdate"] <= cutoff],
            key=lambda s: s["sortpubdate"],
            reverse=True,
        )
        ll = f"LitVar2 {current}"
        claim(
            "Literature",
            f"LitVar2 links {len(pmids)} PubMed articles to {current}"
            + (
                "."
                if latest_mode
                else f"; {len(in_scope)} of the {len(dated)} with a PubMed date were published on or before {as_of}."
            ),
            [Citation(lit["record_id"], "pmids", None, ll)],
        )
        gene_hint = (genes_vep[:1] or dd.get("genes", [])[:1] or [""])[0]
        ranked = sorted(
            in_scope,
            key=lambda s: (
                -(current in s["title"].lower())
                - (bool(gene_hint) and gene_hint.lower() in s["title"].lower()),
                [-ord(c) for c in s["sortpubdate"]],
            ),
        )
        top = ranked[:max_abstracts]
        abstracts = self.s.pubmed_abstracts([s["pmid"] for s in top]) if top else {}
        for s in top:
            rid = f"PMID:{s['pmid']}"
            add(
                {
                    "record_id": rid,
                    "source": "PubMed",
                    "source_version": "PubMed E-utilities",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/",
                    "retrieved_at": s["retrieved_at"],
                    "status": 200,
                    "data": {
                        **{k: s[k] for k in ("pmid", "title", "journal", "pubdate", "sortpubdate")},
                        "abstract": abstracts.get(s["pmid"], ""),
                    },
                }
            )
            title = s["title"].rstrip(".")
            claim(
                "Literature",
                f"Linked article: \"{title}\" ({s['journal']}, {s['sortpubdate'][:4]}).",
                [
                    Citation(rid, "title", s["title"], f"PMID:{s['pmid']}"),
                    Citation(rid, "journal", s["journal"], f"PMID:{s['pmid']}", shown=s["journal"]),
                ],
            )
        literature = {
            "n_pmids": len(pmids),
            "n_in_scope": len(in_scope),
            "pmids_in_scope": [s["pmid"] for s in in_scope],
            "top": [s["pmid"] for s in top],
        }

        # ---------------- reconcile
        conflicts = self._reconcile(
            alleles,
            clinvar_alleles,
            genes_vep,
            dd.get("genes", []),
            gwas_summary,
            db,
            vep,
            gw,
            claim,
            current,
        )

        # ---------------- trait versus disease, bottom line
        bottom = self._bottom_line(
            current,
            alleles,
            clinvar_alleles,
            gwas_summary,
            somatic,
            conflicts,
            literature,
            claim,
            cur_snap,
            gw,
            lit,
        )

        return {
            "rsid_query": rsid,
            "rsid_current": current,
            "as_of": as_of,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "genes": {
                "vep": genes_vep,
                "dbsnp": dd.get("genes", []),
                "clinvar": sorted({c["row"]["gene_symbol"] for c in clinvar_alleles}),
                "gwas": gwas_summary["mapped_genes"],
            },
            "variant_type": dd.get("variant_type"),
            "alleles": [{k: v for k, v in a.items()} for a in alleles],
            "clinvar": [
                {
                    "variation_id": c["vid"],
                    "allele_index": c["allele"]["index"] if c["allele"] else None,
                    "name": c["row"]["name"],
                    "classification": c["row"]["clnsig"],
                    "review_status": c["row"]["review_status"],
                    "conditions": split_conditions(c["row"]["phenotype_list"]),
                    "origin": c["row"]["origin"],
                    "n_submitters": c["row"]["n_submitters"],
                    "history": c["history"],
                    "submission_classes": dict(Counter(s["clnsig"] for s in c["subs"])),
                }
                for c in clinvar_alleles
            ],
            "clinvar_release": cur_snap,
            "somatic": somatic,
            "gwas": gwas_summary,
            "literature": literature,
            "conflicts": conflicts,
            "bottom_line": bottom,
            "notes": notes,
            "claims": [c.to_dict() for c in claims],
            "records": records,
        }

    # ------------------------------------------------------------------ frequency helpers
    def _freq_v2(self, a: dict, hits: list[dict], mv: dict, claim, rsid: str) -> None:
        idx = None
        for i, h in enumerate(hits):
            v = h.get("vcf") or {}
            if (v.get("ref"), v.get("alt")) == (a["ref"], a["alt"]):
                idx = i
        if (
            idx is None
            and len(hits) == 1
            and a["index"] == 1
            and len(a["ref"]) == len(a["alt"]) == 1
        ):
            v = hits[0].get("vcf") or {}
            if v.get("alt") == a["alt"]:
                idx = 0
        label = "gnomAD v2.1.1"
        if idx is None:
            claim(
                "Frequency",
                f"Alternate allele {a['index']} ({a['ref']}>{a['alt']}) has no gnomAD v2.1.1 record among the {len(hits)} MyVariant.info hit{'s' if len(hits) != 1 else ''} for {rsid}.",
                [Citation(mv["record_id"], "hits", None, label)],
            )
            a["frequency"]["gnomad_v2"] = None
            return
        h = hits[idx]
        best = None
        for part in ("gnomad_exome", "gnomad_genome"):
            blk = h.get(part)
            if (
                isinstance(blk, dict)
                and isinstance(blk.get("af"), dict)
                and blk["af"].get("af") is not None
            ):
                an = (blk.get("an") or {}).get("an", 0) or 0
                if best is None or an > best[1]:
                    best = (part, an)
        if best is None:
            claim(
                "Frequency",
                f"Alternate allele {a['index']} ({a['ref']}>{a['alt']}) was not observed in gnomAD v2.1.1 exomes or genomes.",
                [Citation(mv["record_id"], f"hits.{idx}._id", h.get("_id"), label)],
            )
            a["frequency"]["gnomad_v2"] = {"dataset": None, "overall": 0.0, "populations": {}}
            return
        part = best[0]
        af = h[part]["af"]
        ds = "exomes" if part == "gnomad_exome" else "genomes"
        pops = {p: float(af[f"af_{p}"]) for p in GNOMAD_V2_POPS if af.get(f"af_{p}") is not None}
        overall = float(af["af"])
        a["frequency"]["gnomad_v2"] = {
            "dataset": ds,
            "overall": overall,
            "populations": pops,
            "an": best[1],
        }
        base = f"hits.{idx}.{part}.af"
        claim(
            "Frequency",
            f"In gnomAD v2.1.1 {ds}, alternate allele {a['index']} ({a['ref']}>{a['alt']}) has an overall allele frequency of {fmt_af(overall)}.",
            [
                Citation(
                    mv["record_id"], f"{base}.af", overall, f"{label} {ds}", shown=fmt_af(overall)
                )
            ],
        )
        if pops:
            txt = ", ".join(f"{POP_NAMES[p]} {fmt_af(v)}" for p, v in pops.items())
            claim(
                "Frequency",
                f"Per-ancestry gnomAD v2.1.1 frequencies for allele {a['index']}: {txt}.",
                [
                    Citation(mv["record_id"], f"{base}.af_{p}", v, f"{label} {ds}", shown=fmt_af(v))
                    for p, v in pops.items()
                ],
            )
            self._differentiation(
                a, pops, claim, mv["record_id"], base, f"{label} {ds}", "af_{}", "gnomad_v2"
            )

    def _freq_v4(self, a: dict, g4: dict, claim) -> None:
        var = g4["data"].get("variant")
        label = "gnomAD v4.1"
        if not var:
            claim(
                "Frequency",
                f"gnomAD v4.1 has no record for {a['gnomad_v4_id']} (alternate allele {a['index']}).",
                [Citation(g4["record_id"], "variant", None, label)],
            )
            a["frequency"]["gnomad_v4"] = None
            return
        parts = [p for p in ("exome", "genome") if var.get(p)]
        num = [f"variant.{p}.ac" for p in parts]
        den = [f"variant.{p}.an" for p in parts]
        ac = sum(var[p]["ac"] for p in parts)
        an = sum(var[p]["an"] for p in parts)
        if an == 0:
            a["frequency"]["gnomad_v4"] = None
            return
        overall = ac / an
        pops, cits = {}, []
        for pop in GNOMAD_V4_POPS:
            pn, pd_, pac, pan = [], [], 0, 0
            for p in parts:
                for j, e in enumerate(var[p]["populations"]):
                    if e["id"] == pop:
                        pn.append(f"variant.{p}.populations.{j}.ac")
                        pd_.append(f"variant.{p}.populations.{j}.an")
                        pac += e["ac"]
                        pan += e["an"]
            if pan > 0:
                pops[pop] = pac / pan
                cits.append(
                    Citation(
                        g4["record_id"],
                        "",
                        pops[pop],
                        label,
                        mode="ratio_of_sums",
                        shown=fmt_af(pops[pop]),
                        args=[pn, pd_],
                    )
                )
        a["frequency"]["gnomad_v4"] = {"overall": overall, "ac": ac, "an": an, "populations": pops}
        claim(
            "Frequency",
            f"In gnomAD v4.1 (exomes and genomes combined), alternate allele {a['index']} ({a['ref']}>{a['alt']}) has an overall allele frequency of {fmt_af(overall)} ({ac:,} of {an:,} alleles).",
            [
                Citation(
                    g4["record_id"],
                    "",
                    overall,
                    label,
                    mode="ratio_of_sums",
                    shown=fmt_af(overall),
                    args=[num, den],
                )
            ],
        )
        if pops:
            txt = ", ".join(f"{POP_NAMES[p]} {fmt_af(v)}" for p, v in pops.items())
            claim(
                "Frequency",
                f"Per-ancestry gnomAD v4.1 frequencies for allele {a['index']}: {txt}.",
                cits,
            )
            mx, mn = max(pops, key=pops.get), min(pops, key=pops.get)
            if _large_difference(pops):
                a["frequency"]["gnomad_v4"]["differentiated"] = True
                keep = [c for c in cits if c.shown in (fmt_af(pops[mx]), fmt_af(pops[mn]))]
                claim(
                    "Frequency",
                    f"Population differentiation flag (gnomAD v4.1): allele {a['index']} frequency ranges from {fmt_af(pops[mn])} in {POP_NAMES[mn]} to {fmt_af(pops[mx])} in {POP_NAMES[mx]}, a large between-ancestry difference.",
                    keep,
                )

    def _differentiation(self, a, pops, claim, rid, base, label, key_fmt, store) -> None:
        if not _large_difference(pops):
            return
        mx, mn = max(pops, key=pops.get), min(pops, key=pops.get)
        a["frequency"][store]["differentiated"] = True
        claim(
            "Frequency",
            f"Population differentiation flag ({label}): allele {a['index']} frequency ranges from {fmt_af(pops[mn])} in {POP_NAMES[mn]} to {fmt_af(pops[mx])} in {POP_NAMES[mx]}, a large between-ancestry difference.",
            [
                Citation(
                    rid, f"{base}.{key_fmt.format(mn)}", pops[mn], label, shown=fmt_af(pops[mn])
                ),
                Citation(
                    rid, f"{base}.{key_fmt.format(mx)}", pops[mx], label, shown=fmt_af(pops[mx])
                ),
            ],
        )

    # ------------------------------------------------------------------ ClinVar
    def _clinvar_claims(
        self, cvs: list[dict], snap: str, claim, rsid: str, latest_mode: bool
    ) -> None:
        for c in cvs:
            row, rec = c["row"], c["record"]
            vid = c["vid"]
            lab = (
                f"ClinVar VCV{int(vid):09d} ({snap} release)"
                if str(vid).isdigit()
                else f"ClinVar allele {row['allele_id']} ({snap} release)"
            )
            who = (
                f"alternate allele {c['allele']['index']}"
                if c["allele"]
                else "an allele not matched to the dbSNP allele list"
            )
            conds = split_conditions(row["phenotype_list"])
            shown_c = "; ".join(conds[:4]) + (
                f"; and {len(conds) - 4} more" if len(conds) > 4 else ""
            )
            cits = [
                Citation(rec["record_id"], "clnsig", row["clnsig"], lab, shown=row["clnsig"]),
                Citation(
                    rec["record_id"],
                    "review_status",
                    row["review_status"],
                    lab,
                    shown=row["review_status"],
                ),
                Citation(rec["record_id"], "name", row["name"], lab, shown=row["name"]),
                Citation(
                    rec["record_id"],
                    "n_submitters",
                    row["n_submitters"],
                    lab,
                    shown=str(row["n_submitters"]),
                ),
            ]
            cits += [
                Citation(rec["record_id"], "phenotype_list", x, lab, mode="contains", shown=x)
                for x in conds[:4]
            ]
            cond_txt = f" for: {shown_c}" if conds else " with no specific condition named"
            claim(
                "Disease relevance (germline)",
                f"ClinVar lists {row['name']} ({who}) with germline classification '{row['clnsig']}'{cond_txt}; review status '{row['review_status']}', {row['n_submitters']} submitter(s), allele origin '{row['origin']}'.",
                cits
                + [Citation(rec["record_id"], "origin", row["origin"], lab, shown=row["origin"])],
            )
            hist = c["history"]
            if len(hist) > 1:
                seq = "; ".join(f"{h['snapshot']}: {h['clnsig']}" for h in hist)
                changed = len({h["clnsig"] for h in hist}) > 1
                claim(
                    "Disease relevance (germline)",
                    f"Classification history across archived ClinVar releases for this allele: {seq}. The classification {'changed' if changed else 'did not change'} over these releases.",
                    [
                        Citation(
                            rec["record_id"],
                            f"history.{i}.clnsig",
                            h["clnsig"],
                            lab,
                            shown=h["clnsig"],
                        )
                        for i, h in enumerate(hist)
                    ],
                )
            subs = c["subs"]
            if subs:
                cnt = Counter(s["clnsig"] for s in subs)
                txt = ", ".join(
                    f"{n} '{k}'" if k != "-" else f"{n} without a germline classification"
                    for k, n in cnt.most_common()
                )
                claim(
                    "Disease relevance (germline)",
                    f"Individual ClinVar submissions for this allele in the {snap} release: {txt}.",
                    [
                        Citation(
                            rec["record_id"],
                            "",
                            n,
                            lab,
                            mode="count_eq",
                            shown=str(n),
                            args=["submissions", "clnsig", k],
                        )
                        for k, n in cnt.most_common()
                    ],
                )

    # ------------------------------------------------------------------ somatic
    def _somatic_claims(self, alleles, cvs, vep, mv, add, claim, somatic, rsid, notes) -> None:
        sec = "Somatic evidence"
        vl = f"Ensembl VEP {rsid}"
        signals = somatic["signals"]
        cos = vep["data"].get("cosmic_ids") or []
        if cos:
            shown = ", ".join(cos[:5])
            claim(
                sec,
                f"Ensembl VEP reports {len(cos)} co-located COSMIC somatic mutation record(s) at this position ({shown}).",
                [
                    Citation(vep["record_id"], f"cosmic_ids.{i}", cos[i], vl, shown=cos[i])
                    for i in range(min(5, len(cos)))
                ],
            )
            signals.append("cosmic_colocated")
        for c in cvs:
            row, rec = c["row"], c["record"]
            lab = (
                f"ClinVar VCV{int(c['vid']):09d} ({LATEST} release)"
                if str(c["vid"]).isdigit()
                else "ClinVar"
            )
            onc = row.get("oncogenicity") or ""
            sci = row.get("somatic_clinical_impact") or ""
            if onc not in ("", "-") and not onc.lower().startswith("no classification"):
                claim(
                    sec,
                    f"ClinVar gives {row['name']} a somatic oncogenicity classification of '{onc}'.",
                    [Citation(rec["record_id"], "oncogenicity", onc, lab, shown=onc)],
                )
                signals.append("clinvar_oncogenicity")
            if sci not in ("", "-") and not sci.lower().startswith("no classification"):
                claim(
                    sec,
                    f"ClinVar gives {row['name']} a somatic clinical impact classification of '{sci}'.",
                    [Citation(rec["record_id"], "somatic_clinical_impact", sci, lab, shown=sci)],
                )
                signals.append("clinvar_somatic_impact")
            if "somatic" in (row.get("origin") or ""):
                signals.append("clinvar_origin_somatic")
        for a in alleles:
            v = a.get("vep") or {}
            short = short_protein_change(v.get("hgvsp"))
            if not (short and v.get("gene_symbol")):
                continue
            try:
                cv = add(self.s.civic(v["gene_symbol"], short))
            except Exception as err:
                notes.append(f"CIViC lookup failed for {v['gene_symbol']} {short}: {err}")
                continue
            vs = cv["data"].get("variants") or []
            if vs:
                n = vs[0]["n_evidence_items"]
                claim(
                    sec,
                    f"CIViC has a curated entry for {v['gene_symbol']} {short} (alternate allele {a['index']}) with {n} evidence item(s) across {vs[0]['n_molecular_profiles']} molecular profile(s).",
                    [
                        Citation(
                            cv["record_id"],
                            "variants.0.n_evidence_items",
                            n,
                            f"CIViC variant {vs[0]['id']}",
                            shown=str(n),
                        ),
                        Citation(
                            cv["record_id"],
                            "variants.0.n_molecular_profiles",
                            vs[0]["n_molecular_profiles"],
                            f"CIViC variant {vs[0]['id']}",
                            shown=str(vs[0]["n_molecular_profiles"]),
                        ),
                    ],
                )
                a["civic"] = vs[0]
                if n >= 5:
                    signals.append("civic_evidence")
        for i, h in enumerate(mv["data"].get("hits", [])):
            d = h.get("docm")
            if isinstance(d, dict) and d.get("disease"):
                dis = d["disease"] if isinstance(d["disease"], str) else ", ".join(d["disease"][:3])
                first = d["disease"] if isinstance(d["disease"], str) else d["disease"][0]
                path = (
                    f"hits.{i}.docm.disease"
                    if isinstance(d["disease"], str)
                    else f"hits.{i}.docm.disease.0"
                )
                claim(
                    sec,
                    f"The Database of Curated Mutations (DoCM) lists {h.get('_id')} for: {dis}.",
                    [
                        Citation(
                            mv["record_id"], path, first, "DoCM via MyVariant.info", shown=first
                        )
                    ],
                )
                signals.append("docm")
        strong = {"clinvar_oncogenicity", "clinvar_somatic_impact", "civic_evidence", "docm"}
        somatic["is_somatic_hotspot"] = (
            bool(strong & set(signals)) and "cosmic_colocated" in signals
        )
        if not signals:
            claim(
                sec,
                "No somatic evidence was retrieved: Ensembl VEP reports no co-located COSMIC record for this variant.",
                [Citation(vep["record_id"], "cosmic_ids", [], vl)],
            )

    # ------------------------------------------------------------------ GWAS
    def _gwas_claims(self, assoc, assoc_all, gw, pm, add, claim, rsid, as_of, latest_mode) -> dict:
        sec = "Trait associations (GWAS)"
        gl = f"GWAS Catalog {rsid}"
        studies = sorted({a["accession_id"] for a in assoc if a.get("accession_id")})
        mapped = sorted({g for a in assoc for g in (a.get("mapped_genes") or [])})
        sig = [a for a in assoc if a.get("p_value") is not None and assoc_p(a)[0] <= -7.3]
        if not assoc_all:
            claim(
                sec,
                f"The GWAS Catalog lists no association for {rsid}.",
                [Citation(gw["record_id"], "associations", [], gl)],
            )
        else:
            scope = (
                ""
                if latest_mode
                else f" Of these, {len(assoc)} were published on or before {as_of}."
            )
            claim(
                sec,
                f"The GWAS Catalog lists {len(assoc_all)} association record(s) for {rsid}.{scope}",
                [
                    Citation(
                        gw["record_id"],
                        f"associations.{len(assoc_all) - 1}.association_id",
                        assoc_all[-1]["association_id"],
                        gl,
                    )
                ],
            )
        best: dict[str, dict] = {}
        for a in sig:
            traits = [
                t.get("efo_trait") for t in (a.get("efo_traits") or []) if t.get("efo_trait")
            ] or (a.get("reported_trait") or [])
            for t in traits[:1]:
                if t not in best or assoc_p(a)[0] < assoc_p(best[t])[0]:
                    best[t] = a
        top = sorted(best.items(), key=lambda kv: assoc_p(kv[1])[0])[:6]
        for trait, a in top:
            i = a["_index"]
            pmid = str(a.get("pubmed_id"))
            cits = [
                Citation(
                    gw["record_id"],
                    f"associations.{i}.accession_id",
                    a["accession_id"],
                    f"GWAS Catalog {a['accession_id']}",
                    shown=a["accession_id"],
                ),
                Citation(
                    gw["record_id"],
                    f"associations.{i}.pvalue_exponent",
                    a.get("pvalue_exponent"),
                    f"GWAS Catalog {a['accession_id']}",
                    shown=assoc_p(a)[1],
                ),
            ]
            if pmid in pm:
                rid = f"PMID:{pmid}"
                add(
                    {
                        "record_id": rid,
                        "source": "PubMed",
                        "source_version": "PubMed E-utilities",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "retrieved_at": pm[pmid]["retrieved_at"],
                        "status": 200,
                        "data": {
                            k: pm[pmid][k]
                            for k in ("pmid", "title", "journal", "pubdate", "sortpubdate")
                        },
                    }
                )
                cits.append(Citation(rid, "pmid", pmid, f"PMID:{pmid}"))
            eff = (a.get("snp_effect_allele") or [""])[0]
            eff_txt = f", effect allele {eff}" if eff and not eff.endswith("?") else ""
            claim(
                sec,
                f"Genome-wide significant association with {trait} (p = {assoc_p(a)[1]}, study {a['accession_id']}{eff_txt}).",
                cits,
            )
        return {
            "n_associations_total": len(assoc_all),
            "n_associations_in_scope": len(assoc),
            "n_genome_wide_significant": len(sig),
            "n_studies": len(studies),
            "mapped_genes": mapped,
            "top_traits": [
                {
                    "trait": t,
                    "p_value": assoc_p(a)[1],
                    "accession": a["accession_id"],
                    "pmid": a.get("pubmed_id"),
                }
                for t, a in top
            ],
            "n_distinct_traits_significant": len(best),
            "haplotype_associations": sum(1 for a in assoc if a.get("multi_snp_haplotype")),
        }

    # ------------------------------------------------------------------ reconcile
    def _reconcile(
        self, alleles, cvs, genes_vep, genes_dbsnp, gwas, db, vep, gw, claim, rsid
    ) -> list[dict]:
        sec = "Conflicts and caveats"
        out: list[dict] = []
        # 1. pathogenic claim versus population frequency (BA1-like)
        for c in cvs:
            a = c["allele"]
            if not a or _norm_class(c["row"]["clnsig"]) != "PLP":
                continue
            for key, label in (("gnomad_v4", "gnomAD v4.1"), ("gnomad_v2", "gnomAD v2.1.1")):
                f = (a.get("frequency") or {}).get(key)
                if not f or not f.get("populations"):
                    continue
                mx = max(f["populations"], key=f["populations"].get)
                if f["populations"][mx] > BA1_THRESHOLD:
                    rec = c["record"]
                    lab = (
                        f"ClinVar VCV{int(c['vid']):09d}" if str(c["vid"]).isdigit() else "ClinVar"
                    )
                    claim(
                        sec,
                        f"Frequency conflict: allele {a['index']} is classified '{c['row']['clnsig']}' in ClinVar, yet its {label} frequency in {POP_NAMES[mx]} is {fmt_af(f['populations'][mx])}, above the 0.05 ACMG/AMP BA1 stand-alone benign threshold; a frequency this high is expected only for recessive, low-penetrance or balanced-selection alleles and should be interpreted in that light.",
                        [
                            Citation(
                                rec["record_id"],
                                "clnsig",
                                c["row"]["clnsig"],
                                lab,
                                shown=c["row"]["clnsig"],
                            ),
                            Citation(rec["record_id"], "name", c["row"]["name"], label),
                        ],
                    )
                    out.append(
                        {
                            "type": "pathogenic_vs_frequency",
                            "allele": a["index"],
                            "population": mx,
                            "af": f["populations"][mx],
                            "dataset": label,
                        }
                    )
                    break
        # 2. conflicting submitters
        for c in cvs:
            row, rec = c["row"], c["record"]
            lab = f"ClinVar VCV{int(c['vid']):09d}" if str(c["vid"]).isdigit() else "ClinVar"
            classes = Counter(_norm_class(s["clnsig"]) for s in c["subs"])
            groups = {k for k in classes if k in ("PLP", "VUS", "BLB")}
            agg_conflict = _norm_class(row["clnsig"]) == "CONFLICT"
            if len(groups) > 1 or agg_conflict:
                detail = ", ".join(
                    f"{classes[k]} {n}"
                    for k, n in (
                        ("PLP", "pathogenic or likely pathogenic"),
                        ("VUS", "uncertain significance"),
                        ("BLB", "benign or likely benign"),
                    )
                    if classes.get(k)
                )
                cits = [
                    Citation(rec["record_id"], "clnsig", row["clnsig"], lab, shown=row["clnsig"])
                ]
                if c["subs"]:
                    cits.append(Citation(rec["record_id"], "submissions", None, lab))
                claim(
                    sec,
                    f"Submitter conflict: ClinVar submissions for {row['name']} disagree"
                    + (f" ({detail})" if detail else "")
                    + f"; the aggregate classification is '{row['clnsig']}'.",
                    cits,
                )
                out.append(
                    {
                        "type": "conflicting_submitters",
                        "variation_id": c["vid"],
                        "classes": dict(classes),
                        "aggregate_conflict": agg_conflict,
                    }
                )
        # 3. alleles of one rsID with different clinical meaning
        by_class = {
            c["allele"]["index"]: _norm_class(c["row"]["clnsig"]) for c in cvs if c["allele"]
        }
        if len(set(by_class.values())) > 1:
            txt = "; ".join(
                f"allele {i}: {next(c['row']['clnsig'] for c in cvs if c['allele'] and c['allele']['index'] == i)}"
                for i in sorted(by_class)
            )
            claim(
                sec,
                f"Allele-specific interpretation: the alternate alleles of {rsid} carry different ClinVar classifications ({txt}), so the rsID alone does not determine clinical meaning.",
                [
                    Citation(
                        c["record"]["record_id"],
                        "clnsig",
                        c["row"]["clnsig"],
                        f"ClinVar VCV{int(c['vid']):09d}" if str(c["vid"]).isdigit() else "ClinVar",
                        shown=c["row"]["clnsig"],
                    )
                    for c in cvs
                    if c["allele"]
                ],
            )
            out.append({"type": "allele_specific_classification", "classes": by_class})
        # 4. discordant gene assignment across sources
        sets = {
            "Ensembl VEP": set(genes_vep),
            "ClinVar": {
                g for c in cvs for g in re.split(r"[;,]", c["row"]["gene_symbol"]) if g and g != "-"
            },
            "GWAS Catalog": set(gwas["mapped_genes"]),
        }
        # uncharacterised LOC loci and antisense/readthrough labels are not treated as disagreements
        sets = {
            k: {g for g in v if not re.match(r"^(LOC\d+|LINC\d+)$", g) and "-AS" not in g}
            for k, v in sets.items()
        }
        sets = {k: v for k, v in sets.items() if v}
        names = list(sets)
        disc = [
            (x, y) for i, x in enumerate(names) for y in names[i + 1 :] if not (sets[x] & sets[y])
        ]
        union_differs = len({frozenset(v) for v in sets.values()}) > 1
        if disc or union_differs:
            txt = "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sets.items())
            cits = []
            if "Ensembl VEP" in sets:
                cits.append(
                    Citation(
                        vep["record_id"],
                        "transcript_consequences.0.gene_symbol",
                        None,
                        f"Ensembl VEP {rsid}",
                    )
                )
            if "ClinVar" in sets:
                c0 = cvs[0]
                cits.append(
                    Citation(
                        c0["record"]["record_id"],
                        "gene_symbol",
                        c0["row"]["gene_symbol"],
                        (
                            f"ClinVar VCV{int(c0['vid']):09d}"
                            if str(c0["vid"]).isdigit()
                            else "ClinVar"
                        ),
                    )
                )
            if "GWAS Catalog" in sets:
                cits.append(
                    Citation(
                        gw["record_id"], "associations.0.mapped_genes", None, f"GWAS Catalog {rsid}"
                    )
                )
            kind = "Gene assignment conflict" if disc else "Gene assignment difference"
            claim(sec, f"{kind}: sources name different genes for {rsid} ({txt}).", cits)
            out.append(
                {
                    "type": "discordant_gene_assignment",
                    "strict": bool(disc),
                    "genes": {k: sorted(v) for k, v in sets.items()},
                }
            )
        # 5. classification changed over time
        for c in cvs:
            if len({h["clnsig"] for h in c["history"]}) > 1:
                out.append(
                    {
                        "type": "classification_changed_over_time",
                        "variation_id": c["vid"],
                        "history": [(h["snapshot"], h["clnsig"]) for h in c["history"]],
                    }
                )
        return out

    # ------------------------------------------------------------------ bottom line
    def _bottom_line(
        self, rsid, alleles, cvs, gwas, somatic, conflicts, literature, claim, snap, gw, lit
    ) -> dict:
        classes = [_norm_class(c["row"]["clnsig"]) for c in cvs]
        stars_txt = [c["row"]["review_status"] for c in cvs]
        plp = [c for c in cvs if _norm_class(c["row"]["clnsig"]) == "PLP"]
        strong_review = any(
            s
            in (
                "reviewed by expert panel",
                "practice guideline",
                "criteria provided, multiple submitters, no conflicts",
            )
            for s in stars_txt
        )
        other_terms = " ".join(c["row"]["clnsig"].lower() for c in cvs)
        hard_conflict = any(k["type"] in ("conflicting_submitters",) for k in conflicts)
        n_sig = gwas["n_genome_wide_significant"]
        cits: list[Citation] = []
        for c in cvs[:3]:
            lab = (
                f"ClinVar VCV{int(c['vid']):09d} ({snap} release)"
                if str(c["vid"]).isdigit()
                else "ClinVar"
            )
            cits.append(Citation(c["record"]["record_id"], "clnsig", c["row"]["clnsig"], lab))
        cits.append(Citation(gw["record_id"], "associations", None, f"GWAS Catalog {rsid}"))
        cits.append(Citation(lit["record_id"], "pmids", None, f"LitVar2 {rsid}"))

        sec = "Disease relevance (germline)"
        if not plp:
            if cvs:
                claim(
                    sec,
                    f"No allele of {rsid} has a pathogenic or likely pathogenic germline classification in the ClinVar {snap} release.",
                    [
                        Citation(
                            c["record"]["record_id"],
                            "clnsig",
                            c["row"]["clnsig"],
                            (
                                f"ClinVar VCV{int(c['vid']):09d} ({snap} release)"
                                if str(c["vid"]).isdigit()
                                else "ClinVar"
                            ),
                        )
                        for c in cvs
                    ],
                )
            if n_sig:
                claim(
                    "Trait associations (GWAS)",
                    f"{rsid} is associated with {gwas['n_distinct_traits_significant']} distinct trait(s) at genome-wide significance, and ClinVar holds no pathogenic germline assertion for it, so the evidence supports a trait or risk association and not an established monogenic disease.",
                    [Citation(gw["record_id"], "associations", None, f"GWAS Catalog {rsid}")]
                    + cits[: len(cvs[:3])],
                )

        if somatic.get("is_somatic_hotspot") and plp:
            kind = (
                "recurrent somatic cancer hotspot that also carries germline pathogenic assertions"
            )
        elif somatic.get("is_somatic_hotspot"):
            kind = "variant reported mainly as a somatic cancer mutation"
        elif plp:
            kind = "germline disease variant"
        elif "risk" in other_terms or "low penetrance" in other_terms:
            kind = "low-penetrance risk allele, not a fully penetrant monogenic disease variant"
        elif n_sig:
            kind = "trait-associated common variant without an established monogenic disease role"
            if "drug response" in other_terms:
                kind += " (ClinVar also records a drug-response annotation)"
        elif "drug response" in other_terms:
            kind = "pharmacogenomic or drug-response variant"
        elif "protective" in other_terms or "association" in other_terms:
            kind = "variant with a reported association but no disease classification"
        elif "VUS" in classes or "CONFLICT" in classes:
            kind = "variant of uncertain clinical significance"
        elif "BLB" in classes:
            kind = "benign variant"
        else:
            kind = "variant with little curated evidence"

        expert = any(
            x in ("reviewed by expert panel", "practice guideline")
            for x in (c["row"]["review_status"] for c in plp)
        )
        if (plp and (expert or (strong_review and not hard_conflict))) or (
            not plp and n_sig >= 5 and gwas["n_studies"] >= 3
        ):
            conf, why = "high", (
                "multi-submitter or expert-reviewed ClinVar assertion without submitter conflict"
                if plp
                else f"{n_sig} genome-wide significant associations across {gwas['n_studies']} studies"
            )
        elif plp or n_sig >= 1 or somatic.get("is_somatic_hotspot"):
            conf = "moderate"
            why = (
                "pathogenic assertion with limited review or with conflicting submissions"
                if plp
                else ("replication is limited" if n_sig else "somatic evidence only")
            )
        else:
            conf, why = (
                "low",
                "no pathogenic assertion and no genome-wide significant association were retrieved",
            )

        gene = next(
            (
                a["vep"]["gene_symbol"]
                for a in alleles
                if a.get("vep") and a["vep"].get("gene_symbol")
            ),
            None,
        )
        gene = gene or next(
            (
                g
                for c in cvs
                for g in re.split(r"[;,]", c["row"]["gene_symbol"])
                if g and not g.startswith("LOC")
            ),
            None,
        )
        gene_txt = f" in {gene}" if gene else ""
        if plp:
            conds = split_conditions(plp[0]["row"]["phenotype_list"])
            detail = f" ClinVar links it to {conds[0]}." if conds else ""
        elif gwas["top_traits"]:
            detail = (
                f" Its strongest reported association is with {gwas['top_traits'][0]['trait']}."
            )
        else:
            detail = ""
        material = [
            k
            for k in conflicts
            if k["type"]
            in (
                "pathogenic_vs_frequency",
                "conflicting_submitters",
                "allele_specific_classification",
            )
            or (k["type"] == "discordant_gene_assignment" and k.get("strict"))
        ]
        caveat = " Sources disagree on some points (see conflicts)." if material else ""
        text = f"Bottom line: {rsid}{gene_txt} is best described as: {kind}.{detail}{caveat} Confidence: {conf} ({why})."
        claim("Bottom line", text, cits)
        return {"category": kind, "confidence": conf, "rationale": why, "text": text}


def _large_difference(pops: dict[str, float]) -> bool:
    vals = [v for k, v in pops.items() if k not in ("oth", "remaining")]
    if len(vals) < 2:
        return False
    mx, mn = max(vals), min(vals)
    return (mx >= 0.05 and mx - mn >= 0.2) or (mx >= 0.01 and mx >= 10 * max(mn, 1e-4))


def render_text(d: dict) -> str:
    """Plain-text summary: every sentence carries its bracketed citations."""
    lines = [
        f"VARIANT DOSSIER: {d['rsid_query']}"
        + (f" (current ID {d['rsid_current']})" if d["rsid_current"] != d["rsid_query"] else "")
    ]
    lines.append(f"Evidence as of: {d['as_of']}. Generated {d['generated_at'][:10]}.")
    order = [
        "Identity",
        "Function",
        "Frequency",
        "Disease relevance (germline)",
        "Somatic evidence",
        "Trait associations (GWAS)",
        "Literature",
        "Conflicts and caveats",
        "Bottom line",
    ]
    by: dict[str, list[dict[str, Any]]] = {}
    for c in d["claims"]:
        by.setdefault(c["section"], []).append(c)
    for sec in order:
        if sec not in by:
            if sec == "Conflicts and caveats":
                lines += ["", "CONFLICTS AND CAVEATS", "- (none detected by the automated checks)"]
            continue
        lines.append("")
        lines.append(sec.upper())
        for c in by[sec]:
            labels = []
            for cit in c["citations"]:
                if cit["label"] and cit["label"] not in labels:
                    labels.append(cit["label"])
            lines.append(f"- {c['text']} [{'; '.join(labels)}]")
    if d.get("notes"):
        lines += ["", "PROCESSING NOTES (not evidence)"] + [f"- {n}" for n in d["notes"]]
    return "\n".join(lines) + "\n"
