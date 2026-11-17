"""Automated verification of dossier claims against stored raw records.

Three checks, reported separately:

* citation validity: every citation points at a record that is stored in the dossier and whose
  cited path holds the cited value (or satisfies the cited count / ratio);
* claim-to-record agreement: every value marked as ``shown`` is literally present in the
  sentence text, so the sentence says what the record says;
* coverage: every claim has at least one citation, and the rendered text line carries a
  bracketed citation.

For ``as_of`` dossiers ``temporal_violations`` lists any dated record newer than the cut-off.
"""

from __future__ import annotations

import re

from kggc.variant_dossier.claims import Citation, check_citation

_REL = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")


def verify_dossier(d: dict) -> dict:
    records = d["records"]
    n_cit = n_cit_ok = n_shown = n_shown_ok = 0
    n_claims = len(d["claims"])
    uncited, failures = 0, []
    claims_ok = 0
    for ci, c in enumerate(d["claims"]):
        if not c["citations"]:
            uncited += 1
            failures.append({"claim": ci, "reason": "no citation", "text": c["text"][:120]})
            continue
        ok_all = True
        for cit_d in c["citations"]:
            cit = Citation(**cit_d)
            n_cit += 1
            ok, why = check_citation(cit, records.get(cit.record_id))
            if ok:
                n_cit_ok += 1
            else:
                ok_all = False
                failures.append(
                    {
                        "claim": ci,
                        "record": cit.record_id,
                        "path": cit.path,
                        "reason": why,
                        "text": c["text"][:120],
                    }
                )
            if cit.shown:
                n_shown += 1
                if cit.shown in c["text"]:
                    n_shown_ok += 1
                else:
                    ok_all = False
                    failures.append(
                        {
                            "claim": ci,
                            "record": cit.record_id,
                            "reason": f"shown value {cit.shown!r} not in sentence",
                            "text": c["text"][:120],
                        }
                    )
        claims_ok += ok_all
    return {
        "n_claims": n_claims,
        "n_claims_fully_verified": claims_ok,
        "n_uncited_claims": uncited,
        "n_citations": n_cit,
        "n_citations_valid": n_cit_ok,
        "citation_validity_rate": n_cit_ok / n_cit if n_cit else None,
        "n_shown_values": n_shown,
        "n_shown_values_in_text": n_shown_ok,
        "claim_record_agreement_rate": n_shown_ok / n_shown if n_shown else None,
        "claim_verified_rate": claims_ok / n_claims if n_claims else None,
        "failures": failures[:50],
        "temporal_violations": temporal_violations(d),
    }


def temporal_violations(d: dict) -> list[str]:
    """Dated records newer than as_of (empty for 'latest')."""
    as_of = d["as_of"]
    if as_of == "latest":
        return []
    bad = []
    cited = {c["record_id"] for cl in d["claims"] for c in cl["citations"]}
    for rid, rec in d["records"].items():
        if rid not in cited:
            continue
        if rec["source"] == "PubMed":
            dt = rec["data"].get("sortpubdate", "")
            if dt and dt > as_of:
                bad.append(f"{rid} published {dt}")
        elif rec["source"] == "ClinVar":
            m = _REL.search(rec.get("source_version") or "")
            if m and m.group(1) > as_of:
                bad.append(f"{rid} from ClinVar release dated {m.group(1)}")
        elif rec["source"] in ("gnomAD v4.1", "CIViC"):
            bad.append(f"{rid}: {rec['source']} is not available as of {as_of}")
    return bad


def check_rendered_text(text: str) -> dict:
    """Every evidence line in the plain-text summary must end with a bracketed citation."""
    lines, bad = 0, []
    in_notes = False
    for ln in text.splitlines():
        if ln.startswith("PROCESSING NOTES"):
            in_notes = True
        if not ln.startswith("- ") or in_notes or ln.startswith("- (none detected"):
            continue
        lines += 1
        if not re.search(r"\[[^\[\]]+\]\s*$", ln):
            bad.append(ln[:100])
    return {"n_sentences": lines, "n_without_citation": len(bad), "examples": bad[:5]}
