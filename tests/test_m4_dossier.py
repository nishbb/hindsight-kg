import json
from pathlib import Path

import pytest

from kggc.variant_dossier.build import (
    _large_difference,
    allele_type,
    same_allele,
    spdi_to_vcf,
    split_conditions,
)
from kggc.variant_dossier.claims import Citation, check_citation, resolve_path
from kggc.variant_dossier.sources import short_protein_change
from kggc.variant_dossier.verify import check_rendered_text, verify_dossier

ROOT = Path(__file__).parents[1]
QIAGEN_20 = (
    "rs9807334 rs121913529 rs72921001 rs17822931 rs334 rs987870 rs429358 rs3827760 rs113993960 rs4988235 "
    "rs28934578 rs17004038 rs12913832 rs6025 rs713598 rs7412 rs80357713 rs1426654 rs1799963 rs1801133"
).split()


def test_spdi_to_vcf_and_types():
    assert spdi_to_vcf(5227001, "T", "A") == (5227002, "T", "A")
    assert spdi_to_vcf(100, "TCTT", "T") == (101, "TCTT", "T")  # deletion keeps anchor base
    assert spdi_to_vcf(100, "TCTT", "TCTTCTT") == (101, "T", "TCTT")  # duplication, suffix trimmed
    assert allele_type("T", "A").startswith("single")
    assert allele_type("TCTT", "T") == "deletion"
    assert allele_type("T", "TCTT").startswith("insertion")
    assert allele_type("AG", "TC").startswith("multi")


def test_same_allele_handles_repeat_alignment():
    assert same_allele("ATCT", "A", "TCTT", "T")  # same 3-bp deletion, shifted within the repeat
    assert not same_allele("ATCT", "A", "T", "TCTT")  # deletion versus duplication
    assert not same_allele("C", "T", "C", "G")


def test_helpers():
    assert short_protein_change("ENSP1:p.Gly12Asp") == "G12D"
    assert short_protein_change("ENSP1:p.Phe508del") is None
    assert split_conditions("Cystic fibrosis|not provided|Cystic fibrosis") == ["Cystic fibrosis"]
    assert _large_difference({"afr": 0.9, "eas": 0.01})
    assert not _large_difference({"afr": 0.15, "eas": 0.1})


def test_citation_checker_detects_wrong_values():
    rec = {
        "data": {
            "clnsig": "Pathogenic",
            "subs": [{"c": "P"}, {"c": "P"}, {"c": "B"}],
            "ex": {"ac": 5, "an": 100},
            "ge": {"ac": 5, "an": 100},
        }
    }
    assert check_citation(Citation("r", "clnsig", "Pathogenic"), rec)[0]
    assert not check_citation(Citation("r", "clnsig", "Benign"), rec)[0]
    assert not check_citation(Citation("r", "missing.path", "x"), rec)[0]
    assert not check_citation(Citation("r", "clnsig", "Pathogenic"), None)[0]
    assert check_citation(Citation("r", "", 2, mode="count_eq", args=["subs", "c", "P"]), rec)[0]
    assert not check_citation(Citation("r", "", 3, mode="count_eq", args=["subs", "c", "P"]), rec)[
        0
    ]
    assert check_citation(
        Citation(
            "r", "", 0.05, mode="ratio_of_sums", args=[["ex.ac", "ge.ac"], ["ex.an", "ge.an"]]
        ),
        rec,
    )[0]
    assert not check_citation(
        Citation("r", "", 0.5, mode="ratio_of_sums", args=[["ex.ac"], ["ex.an"]]), rec
    )[0]
    assert resolve_path(rec["data"], "subs.2.c") == (True, "B")


def test_verifier_flags_tampered_dossier():
    d = {
        "as_of": "latest",
        "records": {"A": {"source": "X", "data": {"v": 0.25}}},
        "claims": [
            {
                "section": "s",
                "text": "Frequency is 0.25.",
                "citations": [
                    {
                        "record_id": "A",
                        "path": "v",
                        "value": 0.25,
                        "label": "X",
                        "mode": "eq",
                        "shown": "0.25",
                        "args": None,
                    }
                ],
            }
        ],
    }
    assert verify_dossier(d)["citation_validity_rate"] == 1.0
    d["claims"][0]["text"] = "Frequency is 0.52."  # sentence no longer matches the record
    assert verify_dossier(d)["claim_record_agreement_rate"] == 0.0
    d["records"]["A"]["data"]["v"] = 0.3
    assert verify_dossier(d)["citation_validity_rate"] == 0.0
    d["claims"].append({"section": "s", "text": "Uncited.", "citations": []})
    assert verify_dossier(d)["n_uncited_claims"] == 1


def test_temporal_violation_detected():
    d = {
        "as_of": "2020-01-01",
        "records": {
            "PMID:1": {"source": "PubMed", "data": {"sortpubdate": "2021-03-01", "title": "t"}}
        },
        "claims": [
            {
                "section": "s",
                "text": "x",
                "citations": [
                    {
                        "record_id": "PMID:1",
                        "path": "title",
                        "value": "t",
                        "label": "PMID:1",
                        "mode": "eq",
                        "shown": "",
                        "args": None,
                    }
                ],
            }
        ],
    }
    assert verify_dossier(d)["temporal_violations"]


@pytest.mark.parametrize("folder", ["qiagen_20", "qiagen_20_asof_2020-01-01"])
def test_committed_qiagen_dossiers_are_complete_and_verified(folder):
    base = ROOT / "results" / folder
    if not base.exists():
        pytest.skip("QIAGEN results not generated")
    for rs in QIAGEN_20:
        d = json.loads((base / f"{rs}.json").read_text())
        text = (base / f"{rs}.txt").read_text()
        v = verify_dossier(d)
        assert v["citation_validity_rate"] == 1.0, (rs, v["failures"][:2])
        assert v["claim_record_agreement_rate"] == 1.0, rs
        assert v["n_uncited_claims"] == 0 and not v["temporal_violations"], rs
        assert check_rendered_text(text)["n_without_citation"] == 0, rs
        assert "Bottom line:" in text and "Confidence:" in text
        assert d["alleles"], rs
        assert "—" not in text
    assert (base / "qiagen_20_report.md").exists()
    assert (base / "haplotype_rs429358+rs7412.json").exists()  # APOE pair found automatically
