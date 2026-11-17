import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from kggc.eval.baselines import gbm_cv_oof, gene_folds, task_frame
from kggc.eval.features import ALLOWED_TIME_T, CADD, NEIGHBOR, TABULAR, neighbor_counts
from kggc.eval.metrics import auroc, bootstrap_ci, ece, paired_bootstrap_delta
from kggc.graph.leakage import LeakageError, check_feature_columns
from kggc.models.conformal import SplitConformal, calibrate_by_fold, coverage_report
from kggc.models.gnn import derived_gene_disease, message_edge_dict
from kggc.models.train import TrainConfig, cross_fit


def toy_graph(n_var=240, n_dis=6, n_gene=12, seed=0):
    """Variants with feature 0 set are P/LP for their gene's disease, others B/LB."""
    rng = np.random.default_rng(seed)
    d = HeteroData()
    gene = rng.integers(0, n_gene, n_var)
    is_plp = rng.random(n_var) < 0.5
    x = np.zeros((n_var, 4), dtype=np.float32)
    x[:, 0] = is_plp
    x[:, 1] = ~is_plp
    x[:, 2:] = rng.normal(size=(n_var, 2))
    d["variant"].x = torch.from_numpy(x)
    d["gene"].num_nodes = n_gene
    d["disease"].num_nodes = n_dis
    v = np.arange(n_var)
    dis = gene % n_dis
    d["variant", "in_gene", "gene"].edge_index = torch.tensor(np.stack([v, gene]))
    d["disease", "associates", "gene"].edge_index = torch.tensor(
        np.stack([np.arange(n_gene) % n_dis, np.arange(n_gene)])
    )
    for name, mask in (("plp", is_plp), ("blb", ~is_plp)):
        d["variant", f"assoc_{name}", "disease"].edge_index = torch.tensor(
            np.stack([v[mask], dis[mask]])
        )
    for name in ("vus", "conflict"):
        d["variant", f"assoc_{name}", "disease"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    vtab = pd.DataFrame({"node": v, "variation_id": v + 1000, "gene_id": gene, "origin": "clinvar"})
    return d, vtab


@pytest.mark.parametrize("encoder", ["rgcn", "hgt"])
def test_cross_fit_scores_every_edge_out_of_fold(encoder):
    data, vtab = toy_graph()
    cfg = TrainConfig(
        encoder=encoder,
        hidden=16,
        epochs=120,
        eval_every=10,
        patience=12,
        folds=2,
        lr=0.02,
        dropout=0.0,
    )
    res = cross_fit(data, vtab, cfg, log=lambda *_: None)
    sc = res["scores"]
    n_edges = sum(
        data["variant", f"assoc_{t}", "disease"].edge_index.shape[1] for t in ("plp", "blb")
    )
    assert len(sc) == n_edges  # each assertion edge scored exactly once
    assert not sc.duplicated(["variant_node", "disease_node", "edge_type"]).any()
    # an edge is only scored by the model of the fold that held its variant out
    assert (res["fold_ids"][sc["variant_node"].values] == sc["fold"].values).all()
    for _, grp in sc.groupby("fold"):  # the toy signal is learnable within each held-out fold
        assert auroc((grp["edge_type"] == "plp").astype(int), grp["score"]) > 0.85


def test_variant_never_receives_assertion_messages():
    data, _ = toy_graph(n_var=20)
    static = {("variant", "in_gene", "gene"): data["variant", "in_gene", "gene"].edge_index}
    vg = torch.zeros(20, dtype=torch.long)
    msg = message_edge_dict(static, {"plp": data["variant", "assoc_plp", "disease"].edge_index}, vg)
    for src, rel, dst in msg:
        if "assoc" in rel or "variant_for" in rel:
            assert dst != "variant"


def test_derived_edges_drop_variants_without_gene():
    ei = torch.tensor([[0, 1, 2], [5, 5, 6]])
    vg = torch.tensor([3, 3, -1])
    out = derived_gene_disease(ei, vg)
    assert out.tolist() == [[3], [5]]


def test_split_conformal_coverage():
    rng = np.random.default_rng(0)
    plp, blb = rng.normal(2, 1.5, 4000), rng.normal(-2, 1.5, 4000)
    sc = SplitConformal(alpha=0.1).fit(plp[:2000], blb[:2000])
    out_p, out_b = sc.predict(plp[2000:]), sc.predict(blb[2000:])
    assert abs(out_p["conformal_set"].isin(["plp", "both"]).mean() - 0.9) < 0.03
    assert abs(out_b["conformal_set"].isin(["blb", "both"]).mean() - 0.9) < 0.03
    df = pd.DataFrame(
        {
            "fold": 0,
            "edge_type": ["plp"] * 2000 + ["blb"] * 2000,
            "score": np.r_[plp[2000:], blb[2000:]],
        }
    )
    cal = pd.DataFrame(
        {
            "fold": 0,
            "edge_type": ["plp"] * 2000 + ["blb"] * 2000,
            "score": np.r_[plp[:2000], blb[:2000]],
        }
    )
    rep = coverage_report(calibrate_by_fold(df, cal, 0.1))
    assert 0.87 < rep["coverage_plp"] < 0.93


def test_metrics():
    y = np.r_[np.ones(200), np.zeros(200)]
    rng = np.random.default_rng(0)
    good, noise = y + rng.normal(0, 0.5, 400), rng.normal(size=400)
    p, lo, hi = bootstrap_ci(y, good, auroc, 200)
    assert lo < p < hi and p > 0.9
    d, dlo, dhi, pv = paired_bootstrap_delta(y, noise, good, auroc, 200)
    assert d > 0.3 and dlo > 0 and pv < 0.05
    assert ece(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_gene_grouped_folds_keep_genes_together():
    genes = np.repeat(np.arange(50), 7)
    fold = gene_folds(genes, 5, seed=0)
    assert all(len(set(fold[genes == g])) == 1 for g in range(50))
    d = pd.DataFrame({"f": np.random.default_rng(0).normal(size=350), "y": (np.arange(350) % 2)})
    assert not np.isnan(gbm_cv_oof(d, ["f"], fold)).any()


def test_feature_allow_list_blocks_leaky_columns():
    check_feature_columns(TABULAR + NEIGHBOR, ALLOWED_TIME_T)
    for bad in (CADD, ["latest_clnsig_class"], ["alphamissense"], ["gnomad_v4_af"]):
        with pytest.raises(LeakageError):
            check_feature_columns(TABULAR + bad, ALLOWED_TIME_T)


def test_neighbor_counts_are_leave_one_out():
    vtab = pd.DataFrame(
        {
            "origin": "clinvar",
            "gene_id": [1, 1, 1, 2],
            "clnsig_class_T": ["PLP", "PLP", "VUS", "PLP"],
            "consequence": ["missense"] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "variation_id": [10, 11, 12, 13],
            "gene_id": [1, 1, 1, 2],
            "mondo_id": ["M1", "M1", "M1", "M2"],
            "clnsig_class": ["PLP", "PLP", "VUS", "PLP"],
            "consequence": ["missense"] * 4,
        }
    )
    nc = neighbor_counts(vtab, labels).set_index("variation_id")
    assert (
        nc.loc[10, "gene_n_plp"] == 1
        and nc.loc[12, "gene_n_plp"] == 2
        and nc.loc[13, "gene_n_plp"] == 0
    )
    assert nc.loc[10, "gene_disease_n_plp"] == 1 and nc.loc[13, "disease_n_plp"] == 0


def test_task_frame():
    f = pd.DataFrame({"label": ["downgraded", "stable_pathogenic", "upgraded", "stable_vus"]})
    assert task_frame(f, "downgrade")["y"].tolist() == [1, 0]
    assert task_frame(f, "upgrade")["y"].tolist() == [1, 0]
