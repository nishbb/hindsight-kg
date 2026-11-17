"""Minimal OBO parser for MONDO and HPO (no external dependency)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_XREF_RE = re.compile(r"^xref:\s+(\S+)(?:\s+\{(.*)\})?")
_ISA_RE = re.compile(r"^is_a:\s+(\S+)")


@dataclass
class Term:
    id: str
    name: str = ""
    is_a: list[str] = field(default_factory=list)
    xrefs: list[tuple[str, bool]] = field(default_factory=list)  # (curie, is_equivalent)
    obsolete: bool = False
    replaced_by: str | None = None


def parse_obo(path: str | Path) -> dict[str, Term]:
    terms: dict[str, Term] = {}
    cur: Term | None = None
    in_term = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if cur is not None and in_term:
                    terms[cur.id] = cur
                in_term = line == "[Term]"
                cur = None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur = Term(id=line[4:].strip())
            elif cur is None:
                continue
            elif line.startswith("name: "):
                cur.name = line[6:].strip()
            elif line.startswith("is_obsolete: true"):
                cur.obsolete = True
            elif line.startswith("replaced_by: "):
                cur.replaced_by = line[13:].strip()
            elif line.startswith("is_a:"):
                m = _ISA_RE.match(line)
                if m:
                    cur.is_a.append(m.group(1))
            elif line.startswith("xref:"):
                m = _XREF_RE.match(line)
                if m:
                    quals = m.group(2) or ""
                    cur.xrefs.append((m.group(1), "MONDO:equivalentTo" in quals))
    if cur is not None and in_term:
        terms[cur.id] = cur
    return terms


def normalize_curie(curie: str) -> str:
    """Normalize disease identifiers from ClinVar and MONDO into one namespace."""
    c = curie.strip()
    if c.startswith("MONDO:MONDO:"):
        c = c[len("MONDO:") :]
    if c.startswith("Human Phenotype Ontology:"):
        c = c[len("Human Phenotype Ontology:") :]
    pre, _, local = c.partition(":")
    pre_u = pre.upper()
    if pre_u in ("MEDGEN", "UMLS"):
        return f"UMLS:{local}"
    if pre_u == "ORPHANET":
        return f"Orphanet:{local.replace('ORPHA', '')}"
    if pre_u == "OMIM":
        return f"OMIM:{local}"
    if pre_u == "MESH":
        return f"MESH:{local}"
    if pre_u == "MONDO":
        return f"MONDO:{local}"
    if pre_u == "DOID":
        return f"DOID:{local}"
    if pre_u == "EFO":
        return f"EFO:{local}"
    return c


def build_xref_index(terms: dict[str, Term], equivalent_only: bool = True) -> dict[str, str]:
    """Map normalized xref CURIE -> MONDO id. Ambiguous xrefs (to >1 MONDO term) are dropped."""
    hits: dict[str, set[str]] = {}
    for t in terms.values():
        if t.obsolete or not t.id.startswith("MONDO:"):
            continue
        for curie, equiv in t.xrefs:
            if equivalent_only and not equiv:
                continue
            hits.setdefault(normalize_curie(curie), set()).add(t.id)
        hits.setdefault(t.id, set()).add(t.id)
    return {k: next(iter(v)) for k, v in hits.items() if len(v) == 1}
