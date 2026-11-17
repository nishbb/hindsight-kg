import json
from pathlib import Path

import pandas as pd
import pytest

from kggc.data.clinvar import load_variant_summary
from kggc.data.obo import build_xref_index, parse_obo
from kggc.data.sources import SOURCES, T_DATE, time_safe_keys
from kggc.graph.build import (
    KG,
    add_clinvar,
    add_gwas,
    add_hetionet,
    add_hpo,
    add_mondo,
    read_gwas,
    read_hpo_g2p,
    to_heterodata,
)
from kggc.graph.leakage import (
    LeakageError,
    check_dates,
    check_feature_columns,
    check_manifest,
    source_violations,
)

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def tiny():
    mondo = parse_obo(FIX / "mondo_tiny.obo")
    xidx = build_xref_index(mondo, equivalent_only=False)
    kg = KG()
    add_mondo(kg, mondo)
    add_hetionet(kg, FIX / "hetionet_nodes_tiny.tsv", FIX / "hetionet_edges_tiny.sif.gz", xidx)
    add_hpo(kg, parse_obo(FIX / "hp_tiny.obo"), read_hpo_g2p(FIX / "g2p_tiny.txt"))
    t = load_variant_summary(FIX / "variant_summary_2020-01.txt.gz")
    vtab = add_clinvar(kg, t, xidx)
    g = read_gwas(FIX / "gwas_tiny.tsv", T_DATE)
    vtab = add_gwas(kg, g, vtab, xidx)
    return kg, vtab, g, to_heterodata(kg, vtab)


def test_tiny_graph_shape(tiny):
    kg, vtab, g, data = tiny
    assert data["variant"].num_nodes == len(vtab)
    assert data["variant"].x.shape[0] == len(vtab)
    assert ("variant", "assoc_plp", "disease") in data.edge_types
    assert ("variant", "assoc_vus", "disease") in data.edge_types
    # multi-condition P/LP variant 1005 contributes two assertion edges
    n1005 = vtab.loc[vtab.variation_id == 1005, "node"].item()
    ei = data["variant", "assoc_plp", "disease"].edge_index
    assert int((ei[0] == n1005).sum()) == 2
    # somatic (1006) and compound-significance (1007) records are not in the graph
    assert not set(vtab.variation_id) & {1006, 1007}


def test_hetionet_disease_mapping_and_skips(tiny):
    kg, *_ = tiny
    assert kg.report["hetionet_diseases_mapped_to_mondo"] == 1
    assert kg.report["hetionet_diseases_unmapped"] == 1
    assert len(kg.edges[("disease", "associates", "gene")]) == 1  # unmapped disease edge dropped
    assert "side_effect" not in kg.nodes


def test_gwas_filtered_to_before_T(tiny):
    kg, vtab, g, _ = tiny
    assert set(g["rsid"]) == {"rs11", "rs555"}  # p>5e-8 row and 2021 row removed
    assert g["date_added"].max() <= pd.Timestamp(T_DATE)
    # rs11 is an existing ClinVar variant node, rs555 becomes a GWAS-only node
    assert (vtab["origin"] == "gwas").sum() == 1
    assert "RS:rs557" not in kg.nodes["variant"]


def test_every_tiny_source_is_time_safe(tiny):
    *_, data = tiny
    check_manifest({"node_sources": data.node_source, "edge_sources": data.edge_source})


def test_leakage_checker_catches_violations():
    assert source_violations("primekg")  # flagged leaky
    assert source_violations("clinvar_latest")
    assert source_violations("not_a_source")
    assert not source_violations("hetionet+hpo_g2p_T")
    with pytest.raises(LeakageError):
        check_manifest({"edge_sources": {"gene__x__disease": "hetionet+primekg"}})
    with pytest.raises(LeakageError):
        check_manifest({"max_gwas_pub_date": "2021-04-01"})
    with pytest.raises(LeakageError):
        check_dates(["2019-05-01", "2020-06-01"], "literature items")
    check_dates(["2019-05-01", None, "2020-01-02"], "literature items")
    with pytest.raises(LeakageError):
        check_feature_columns(["stars", "alphamissense"], ["stars", "n_submitters"])


def test_registry_dates():
    safe = set(time_safe_keys())
    assert {
        "clinvar_T",
        "hetionet",
        "mondo_T",
        "hpo_obo_T",
        "hpo_g2p_T",
        "gwas_T",
        "gnomad_v2",
    } <= safe
    assert not {"clinvar_latest", "primekg", "gnomad_v4"} & safe
    for k in safe:
        assert SOURCES[k].release_date <= T_DATE


def test_real_graph_no_leakage():
    """Full-graph leakage test: no edge, node feature or dated item post-dates T."""
    mpath = ROOT / "data" / "processed" / "kg_T_leakage_manifest.json"
    if not mpath.exists():
        mpath = ROOT / "results" / "m2" / "leakage_manifest.json"
    if not mpath.exists():
        pytest.skip("time-T graph not built (run scripts/m2_build_kg.py)")
    manifest = json.loads(mpath.read_text())
    check_manifest(manifest)
    vpath = ROOT / "data" / "processed" / "kg_T_variants.parquet"
    if vpath.exists():
        vt = pd.read_parquet(vpath)
        check_dates(
            pd.to_datetime(vt["last_evaluated"], errors="coerce", format="mixed"), "ClinVar records"
        )
        # the variant table must not carry any outcome or post-T column
        assert not [c for c in vt.columns if "latest" in c or c == "label"]
