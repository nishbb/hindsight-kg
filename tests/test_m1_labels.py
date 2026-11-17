from pathlib import Path

import pandas as pd
import pytest

from kggc.data.clinvar import (
    classify,
    consequence_from_name,
    informative_conditions,
    load_variant_summary,
    split_conditions,
)
from kggc.data.history import history_features
from kggc.data.labels import assert_disjoint, build_labels
from kggc.data.obo import build_xref_index, normalize_curie, parse_obo

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def built():
    t = load_variant_summary(FIX / "variant_summary_2020-01.txt.gz")
    latest = load_variant_summary(FIX / "variant_summary.txt.gz")
    xidx = build_xref_index(parse_obo(FIX / "mondo_tiny.obo"))
    labels, rep = build_labels(t, latest, xidx)
    return t, latest, labels, rep


def test_dedup_prefers_grch38(built):
    t, *_ = built
    assert t["variation_id"].is_unique
    assert t.loc[t.variation_id == 1001, "assembly"].item() == "GRCh38"


def test_expected_labels(built):
    *_, labels, _ = built
    got = dict(zip(labels["variation_id"].astype(int), labels["label"]))
    assert got[1001] == "downgraded"  # P -> VUS
    assert got[1002] == "stable_pathogenic"  # LP -> P
    assert got[1003] == "upgraded"  # VUS -> LP
    assert got[1004] == "stable_vus"
    assert got[1009] == "downgraded"  # P -> conflicting (new wording)
    for vid in (
        1005,
        1006,
        1007,
        1008,
        1010,
    ):  # multi-condition, somatic, compound, unmapped, missing
        assert vid not in got or pd.isna(got[vid])


def test_no_variant_in_two_classes(built):
    *_, labels, _ = built
    assert_disjoint(labels)
    bad = labels.copy()
    bad = pd.concat([bad, bad.iloc[[0]].assign(label="stable_vus")])
    with pytest.raises(AssertionError):
        assert_disjoint(bad)


def test_real_label_table_disjoint():
    p = Path(__file__).parents[1] / "data" / "processed" / "labels.parquet"
    if not p.exists():
        pytest.skip("full label table not built (run scripts/m1_labels.py)")
    assert_disjoint(pd.read_parquet(p))


def test_condition_split_both_formats():
    old = split_conditions("MedGen:C1,OMIM:1;MedGen:CN517202", "Dis;not provided")
    new = split_conditions("MONDO:MONDO:1,MedGen:C1||MedGen:C3", "Dis|Other|not provided")
    assert len(informative_conditions(old)) == 1
    assert len(old) == 2 and old[0][1] == ["MedGen:C1", "OMIM:1"]
    assert len(new) == 3 and new[1][1] == []
    assert len(informative_conditions(new)) == 2


def test_normalize_curie():
    assert normalize_curie("MedGen:C0000001") == "UMLS:C0000001"
    assert normalize_curie("Orphanet:ORPHA77") == "Orphanet:77"
    assert normalize_curie("MONDO:MONDO:0000001") == "MONDO:0000001"
    assert normalize_curie("Human Phenotype Ontology:HP:0001250") == "HP:0001250"


def test_xref_index_equivalent_only():
    xidx = build_xref_index(parse_obo(FIX / "mondo_tiny.obo"))
    assert xidx["UMLS:C0000002"] == "MONDO:0000002"  # equivalentTo not first qualifier
    assert "DOID:999" not in xidx  # relatedTo is not used
    assert "OMIM:100009" not in xidx  # obsolete terms excluded


def test_classify_exact_only():
    assert classify("Pathogenic") == "PLP"
    assert classify("Pathogenic, risk factor") == "OTHER"
    assert classify("Conflicting classifications of pathogenicity") == "CONFLICT"
    assert classify("Conflicting interpretations of pathogenicity") == "CONFLICT"


@pytest.mark.parametrize(
    "name,exp",
    [
        ("NM_1(G):c.10A>G (p.Lys4Arg)", "missense"),
        ("NM_1(G):c.20C>T (p.Arg7Ter)", "nonsense"),
        ("NM_1(G):c.17_18del (p.Leu6fs)", "frameshift"),
        ("NM_1(G):c.7G>A (p.Gly3=)", "synonymous"),
        ("NM_1(G):c.100+1G>A", "splice_site"),
        ("NM_1(G):c.100+35G>A", "intronic"),
        ("NM_1(G):c.*35G>A", "utr3"),
    ],
)
def test_consequence(name, exp):
    assert consequence_from_name(name, "single nucleotide variant") == exp


def test_history_without_variation_id_column(built):
    t, *_ = built
    a2v = dict(zip(t["allele_id"], t["variation_id"].astype(int)))
    s18 = load_variant_summary(FIX / "variant_summary_2018-01.txt.gz", allele_to_variation=a2v)
    assert s18["variation_id"].tolist() == [1001]
    h = history_features([s18, t], pd.Series([1001, 1002]))
    h = h.set_index("variation_id")
    assert h.loc[1001, "ever_conflicting_before_T"] == 1
    assert h.loc[1001, "class_changed_before_T"] == 1
    assert h.loc[1002, "n_prior_snapshots_present"] == 1
