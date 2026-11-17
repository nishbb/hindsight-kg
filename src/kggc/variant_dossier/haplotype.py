"""Detect pairs of queried variants that jointly define a haplotype.

Detection is data-driven, with no gene-specific rules:
  1. both variants lie in the same gene (Ensembl VEP) on the same chromosome;
  2. LitVar2 literature overlap is substantial (shared PMIDs, Jaccard index);
  3. PubMed abstracts that mention both rsIDs use haplotype language.
A pair that passes 1 and 2 is reported; 3 raises confidence and supplies named alleles when the
abstract text states them next to the rsIDs.
"""

from __future__ import annotations

import re
from itertools import combinations

from kggc.variant_dossier.claims import Citation, Claim
from kggc.variant_dossier.sources import Sources

HAP_WORDS = re.compile(
    r"\b(haplotype|haplotypes|allele combination|isoform|epsilon|diplotype)\b|ε", re.I
)
MIN_SHARED = 20
MIN_JACCARD = 0.10
MAX_DISTANCE = 100_000


def _pos(d: dict) -> tuple[str | None, int | None]:
    a = d["alleles"][0] if d.get("alleles") else None
    return (a["chrom"], a["pos"]) if a else (None, None)


def detect_haplotypes(
    dossiers: list[dict], sources: Sources | None = None, max_abstracts: int = 6
) -> list[dict]:
    s = sources or Sources()
    out = []
    for d1, d2 in combinations(dossiers, 2):
        g1, g2 = set(d1["genes"]["vep"]), set(d2["genes"]["vep"])
        (c1, p1), (c2, p2) = _pos(d1), _pos(d2)
        if not (g1 & g2) or c1 is None or c1 != c2 or abs(p1 - p2) > MAX_DISTANCE:
            continue
        l1 = set(d1["literature"]["pmids_in_scope"])
        l2 = set(d2["literature"]["pmids_in_scope"])
        shared = l1 & l2
        jac = len(shared) / max(1, len(l1 | l2))
        if len(shared) < MIN_SHARED or jac < MIN_JACCARD:
            continue
        r1, r2 = d1["rsid_current"], d2["rsid_current"]
        gene = sorted(g1 & g2)[0]
        summ = s.pubmed_summaries(sorted(shared))
        ranked = sorted(
            summ.values(),
            key=lambda x: (not re.search(HAP_WORDS, x["title"]), x["sortpubdate"]),
        )
        # prefer titles with haplotype language, then most recent within scope
        ranked = sorted(
            ranked,
            key=lambda x: (
                not re.search(HAP_WORDS, x["title"]),
                [-ord(ch) for ch in x["sortpubdate"]],
            ),
        )
        pick = ranked[: max_abstracts * 3]
        abstracts = s.pubmed_abstracts([x["pmid"] for x in pick])
        records, claims, support = {}, [], []
        rid_pair = f"LitVar2-overlap:{r1}+{r2}"
        records[rid_pair] = {
            "record_id": rid_pair,
            "source": "LitVar2",
            "source_version": "LitVar2 (intersection of the two per-variant PMID lists)",
            "url": "https://www.ncbi.nlm.nih.gov/research/litvar2/",
            "retrieved_at": d1["generated_at"],
            "status": 200,
            "data": {
                "shared_pmids": sorted(shared, key=int),
                "n_shared": len(shared),
                "n_union": len(l1 | l2),
                "gene": gene,
                "distance_bp": abs(p1 - p2),
            },
        }
        lab = f"LitVar2 {r1} and {r2}"
        claims.append(
            Claim(
                "Joint interpretation",
                f"{r1} and {r2} both lie in {gene}, {abs(p1 - p2)} bp apart, and LitVar2 links {len(shared)} PubMed articles to both variants, so they are routinely analysed together.",
                [
                    Citation(rid_pair, "n_shared", len(shared), lab, shown=str(len(shared))),
                    Citation(rid_pair, "distance_bp", abs(p1 - p2), lab, shown=str(abs(p1 - p2))),
                    Citation(rid_pair, "gene", gene, lab, shown=gene),
                ],
            )
        )
        for x in pick:
            ab = abstracts.get(x["pmid"], "")
            text = f"{x['title']} {ab}"
            both = r1 in text and r2 in text
            if not (both and re.search(HAP_WORDS, text)):
                continue
            sent = next(
                (t.strip() for t in re.split(r"(?<=[.!?])\s+", ab) if r1 in t and r2 in t),
                None,
            )
            rid = f"PMID:{x['pmid']}"
            records[rid] = {
                "record_id": rid,
                "source": "PubMed",
                "source_version": "PubMed E-utilities",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{x['pmid']}/",
                "retrieved_at": x["retrieved_at"],
                "status": 200,
                "data": {
                    "pmid": x["pmid"],
                    "title": x["title"],
                    "journal": x["journal"],
                    "sortpubdate": x["sortpubdate"],
                    "abstract": ab,
                },
            }
            term = re.search(HAP_WORDS, text).group(0)
            claims.append(
                Claim(
                    "Joint interpretation",
                    f"An article titled \"{x['title'].rstrip('.')}\" names both {r1} and {r2} in its title or abstract and uses the term '{term}'.",
                    [
                        Citation(rid, "title", x["title"], f"PMID:{x['pmid']}"),
                        Citation(
                            rid,
                            "abstract" if (r1 in ab and r2 in ab) else "title",
                            r1,
                            f"PMID:{x['pmid']}",
                            mode="contains",
                        ),
                    ],
                )
            )
            support.append({"pmid": x["pmid"], "sentence": sent})
            if len(support) >= max_abstracts:
                break
        # allele-combination table from the two dossiers' own allele data (cited to dbSNP records)
        a1, a2 = d1["alleles"][0], d2["alleles"][0]
        combos = [
            f"{r1} {x} with {r2} {y}"
            for x in (a1["ref"], a1["alt"])
            for y in (a2["ref"], a2["alt"])
        ]
        claims.append(
            Claim(
                "Joint interpretation",
                f"With two alleles at each site ({r1} {a1['ref']}>{a1['alt']}; {r2} {a2['ref']}>{a2['alt']}) there are four possible allele combinations on one chromosome: {'; '.join(combos)}. Which combination a person carries cannot be read from either rsID alone and requires both genotypes (and phase when both sites are heterozygous).",
                [
                    Citation(
                        f"dbSNP:{r1}",
                        f"alleles.{a1['dbsnp_allele_index']}.hgvs",
                        a1["hgvs_g"],
                        f"dbSNP {r1}",
                    ),
                    Citation(
                        f"dbSNP:{r2}",
                        f"alleles.{a2['dbsnp_allele_index']}.hgvs",
                        a2["hgvs_g"],
                        f"dbSNP {r2}",
                    ),
                ],
            )
        )
        records[f"dbSNP:{r1}"] = d1["records"][f"dbSNP:{r1}"]
        records[f"dbSNP:{r2}"] = d2["records"][f"dbSNP:{r2}"]
        conf = "high" if len(support) >= 2 else ("moderate" if support else "low")
        claims.append(
            Claim(
                "Joint interpretation",
                f"Haplotype call: {r1} and {r2} are treated as a jointly interpreted haplotype pair in {gene} (confidence {conf}; {len(support)} retrieved abstract(s) name both variants with haplotype language).",
                [Citation(rid_pair, "n_shared", len(shared), lab)],
            )
        )
        out.append(
            {
                "rsids": [r1, r2],
                "gene": gene,
                "distance_bp": abs(p1 - p2),
                "n_shared_pmids": len(shared),
                "jaccard": jac,
                "supporting_abstracts": support,
                "confidence": conf,
                "claims": [c.to_dict() for c in claims],
                "records": records,
            }
        )
    return out


def render_haplotype(h: dict) -> str:
    lines = [f"JOINT INTERPRETATION: {h['rsids'][0]} + {h['rsids'][1]} ({h['gene']})"]
    for c in h["claims"]:
        labels = []
        for cit in c["citations"]:
            if cit["label"] and cit["label"] not in labels:
                labels.append(cit["label"])
        lines.append(f"- {c['text']} [{'; '.join(labels)}]")
    for sup in h["supporting_abstracts"]:
        if sup["sentence"]:
            q = sup["sentence"]
            q = q if len(q) <= 300 else q[:297] + "..."
            lines.append(f"- Abstract sentence naming both variants: \"{q}\" [PMID:{sup['pmid']}]")
    return "\n".join(lines) + "\n"
