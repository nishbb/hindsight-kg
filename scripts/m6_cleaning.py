"""M6 (part 1, H3): does cleaning the time-T graph help downstream gene-disease link prediction?

Prospective target: (gene, disease) pairs that have a germline P/LP ClinVar assertion in the
2026-11 release but no P/LP assertion at T (2020-01), with diseases mapped through the same
MONDO 2019 xref index. A link predictor is trained on the time-T graph and must rank these
new pairs above random non-pairs.

Cleaning arms (each removes P/LP assertion edges from the time-T graph before training):
  original     nothing removed
  gnn_flag     edges whose out-of-fold R-GCN conformal set is {blb} (time-T only, no labels)
  gbm_prosp    the same number of lowest-trust edges by the prospective GBM (fit on 2018 to
               2020 revisions; time-T only)
  random       the same number of random P/LP edges (control)
  oracle       edges that really were downgraded by 2026 (uses future labels; upper bound only)
  downweight   gnn_flag edges kept but moved to a separate 'assoc_plp_lowtrust' relation, so
               the model can learn how much to trust them

Usage: python scripts/m6_cleaning.py [--seeds 0 1 2] [--epochs 120]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from kggc.data.clinvar import informative_conditions, load_variant_summary, split_conditions
from kggc.data.labels import map_condition
from kggc.data.obo import build_xref_index, parse_obo
from kggc.eval.baselines import make_gbm
from kggc.eval.features import tabular_features
from kggc.eval.prospective import PROSPECTIVE_COLS, prospective_training_table
from kggc.models.conformal import calibrate_by_fold
from kggc.models.gnn import HeteroLinkPredictor, all_edge_types, message_edge_dict
from kggc.models.train import static_edges, variant_gene_index
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, seed_everything, write_json


def new_pairs_latest(data, xidx, vtab) -> set[tuple[int, int]]:
    gene_node = {
        int(e.split("::")[1]): i
        for i, e in enumerate(pd.read_parquet(PROC_DIR / "kg_T_nodes" / "gene.parquet")["id"])
    }
    dis_node = {
        m: i
        for i, m in enumerate(pd.read_parquet(PROC_DIR / "kg_T_nodes" / "disease.parquet")["id"])
    }
    lat = load_variant_summary(RAW_DIR / "clinvar" / "variant_summary_2026-11.txt.gz")
    lat = lat[
        (lat["origin_simple"] == "germline") & (lat["clnsig_class"] == "PLP") & (lat["gene_id"] > 0)
    ]
    pairs = set()
    for gid, ids, lst in zip(lat["gene_id"], lat["phenotype_ids"], lat["phenotype_list"]):
        g = gene_node.get(int(gid))
        if g is None:
            continue
        for _, xr in informative_conditions(split_conditions(ids, lst)):
            m = map_condition(xr, xidx)
            if m in dis_node:
                pairs.add((g, dis_node[m]))
    vg = variant_gene_index(data)
    ei = data["variant", "assoc_plp", "disease"].edge_index
    g = vg[ei[0]]
    t_pairs = set(zip(g[g >= 0].tolist(), ei[1][g >= 0].tolist()))
    return pairs - t_pairs, t_pairs


def cleaning_masks(
    data, vtab, seed: int, tag: str = "rgcn", arms: tuple = ("gbm_prosp",)
) -> dict[str, np.ndarray]:
    """Boolean mask over P/LP assertion edges: True = remove."""
    ei = data["variant", "assoc_plp", "disease"].edge_index.numpy()
    node2vid = vtab.set_index("node")["variation_id"]
    dis = pd.read_parquet(PROC_DIR / "kg_T_nodes" / "disease.parquet")["id"]
    e = pd.DataFrame(
        {"variation_id": node2vid.reindex(ei[0]).values, "mondo_id": dis.reindex(ei[1]).values}
    )
    sc = calibrate_by_fold(
        pd.read_parquet(PROC_DIR / f"gnn_scores_{tag}.parquet"),
        pd.read_parquet(PROC_DIR / f"gnn_calibration_{tag}.parquet"),
    )
    sc = sc[sc["edge_type"] == "plp"].drop_duplicates(["variation_id", "mondo_id"])
    e = e.merge(
        sc[["variation_id", "mondo_id", "conformal_set"]],
        on=["variation_id", "mondo_id"],
        how="left",
    )
    flag = (e["conformal_set"] == "blb").to_numpy()
    k = int(flag.sum())

    lab = pd.read_parquet(PROC_DIR / "labels.parquet")
    ann = pd.read_parquet(PROC_DIR / "annotations_T.parquet")
    down = set(lab.loc[lab["label"] == "downgraded", "variation_id"])
    oracle = e["variation_id"].isin(down).to_numpy()

    # prospective GBM trust for every P/LP edge in the label table (others: never removed)
    gbm = np.zeros(len(e), dtype=bool)
    if "gbm_prosp" in arms:  # needs the 2018 annotation pull, skip when unused
        tr = prospective_training_table()
        tr = tr[tr["label"].isin(["downgraded", "stable_pathogenic"])]
        cols = PROSPECTIVE_COLS
        m = make_gbm(0).fit(tr[cols].to_numpy(float), (tr["label"] == "downgraded").to_numpy())
        cur = tabular_features(lab[lab["clnsig_class"] == "PLP"], ann)
        risk = pd.Series(
            m.predict_proba(cur[cols].to_numpy(float))[:, 1], index=cur["variation_id"]
        )
        r = e["variation_id"].map(risk).fillna(-1).to_numpy()
        gbm[np.argsort(-r)[:k]] = True
    rng = np.random.default_rng(seed)
    rnd = np.zeros(len(e), dtype=bool)
    rnd[rng.choice(len(e), k, replace=False)] = True
    return {
        "original": np.zeros(len(e), bool),
        "gnn_flag": flag,
        "gbm_prosp": gbm,
        "random": rnd,
        "oracle": oracle,
        "downweight": flag,
    }


def train_eval(
    data, arm: str, remove: np.ndarray, pos_eval, t_pairs, seed: int, epochs: int, log=print
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    static = static_edges(data, [])
    vg = variant_gene_index(data)
    vx = data["variant"].x
    ei = {
        t: data["variant", f"assoc_{t}", "disease"].edge_index
        for t in ("plp", "blb", "vus", "conflict")
    }
    rm = torch.from_numpy(remove)
    LOW_W = 0.3  # loss weight for gene-disease pairs supported only by low-trust edges
    if arm == "downweight":
        # flagged edges stay in the graph under their own relation and their pairs stay
        # positives, but pairs supported only by flagged edges get weight LOW_W
        sup_edges, low_flag = ei["plp"], rm
        static = dict(static)
        static[("variant", "assoc_plp_lowtrust", "disease")] = torch.zeros((2, 0), dtype=torch.long)
    else:
        sup_edges, low_flag = ei["plp"][:, ~rm], torch.zeros(int((~rm).sum()), dtype=torch.bool)
    plp = sup_edges[:, ~low_flag]  # edges passed as normal assoc_plp messages
    # training pairs = gene-disease pairs supported by the retained P/LP edges
    g = vg[sup_edges[0]]
    ok = g >= 0
    nd = data["disease"].num_nodes
    pair_id = g[ok] * nd + sup_edges[1][ok]
    uniq, inv = torch.unique(pair_id, return_inverse=True)
    edges_ok, low_ok = sup_edges[:, ok], low_flag[ok]
    pairs = torch.stack([uniq // nd, uniq % nd])
    strong = torch.zeros(len(uniq), dtype=torch.bool)
    strong[inv[~low_ok]] = True
    pair_w = torch.where(strong, 1.0, LOW_W)
    no_gene = sup_edges[:, ~ok]
    genes_cv = torch.unique(vg[vg >= 0])
    dis_cand = torch.unique(torch.cat([e[1] for e in ei.values()]))

    num_nodes = {nt: int(data[nt].num_nodes) for nt in data.node_types}
    model = HeteroLinkPredictor(
        num_nodes,
        vx.shape[1],
        all_edge_types(list(static)),
        hidden=64,
        layers=2,
        encoder="rgcn",
        dropout=0.2,
    )
    dec = torch.nn.Sequential(torch.nn.Linear(192, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(
        list(model.parameters()) + list(dec.parameters()), lr=5e-3, weight_decay=1e-5
    )

    def score(h, p):
        a, b = h["gene"][p[0]], h["disease"][p[1]]
        return dec(torch.cat([a, b, a * b], -1)).squeeze(-1)

    def neg(n):
        return torch.stack(
            [
                genes_cv[torch.randint(0, len(genes_cv), (n,))],
                dis_cand[torch.randint(0, len(dis_cand), (n,))],
            ]
        )

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        sup = torch.from_numpy(rng.random(len(uniq)) < 0.3)
        keep_e = ~sup[inv]  # edges of supervised pairs leave the message graph
        msg_plp = torch.cat([edges_ok[:, keep_e & ~low_ok], no_gene[:, ~low_flag[~ok]]], 1)
        st = static
        if arm == "downweight":
            st = dict(static)
            st[("variant", "assoc_plp_lowtrust", "disease")] = edges_ok[:, keep_e & low_ok]
        msg = message_edge_dict(
            st,
            {"plp": msg_plp, "blb": ei["blb"], "vus": ei["vus"], "conflict": ei["conflict"]},
            vg,
        )
        h = model.encode(vx, msg)
        pp = pairs[:, sup]
        lp, ln = score(h, pp), score(h, neg(pp.shape[1]))
        loss = F.binary_cross_entropy_with_logits(
            lp, torch.ones_like(lp), weight=pair_w[sup]
        ) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 40 == 0:
            log(f"[{arm} seed {seed}] ep {ep} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    if arm == "downweight":
        static = dict(static)
        static[("variant", "assoc_plp_lowtrust", "disease")] = sup_edges[:, low_flag]
    with torch.no_grad():
        msg = message_edge_dict(
            static, {"plp": plp, "blb": ei["blb"], "vus": ei["vus"], "conflict": ei["conflict"]}, vg
        )
        h = model.encode(vx, msg)
        pos = torch.tensor(sorted(pos_eval)).T
        # negatives: random (gene with ClinVar variants, disease with assertions) pairs that are not P/LP at T or latest
        g_rng = torch.Generator().manual_seed(1234)
        cand = torch.stack(
            [
                genes_cv[torch.randint(0, len(genes_cv), (pos.shape[1] * 12,), generator=g_rng)],
                dis_cand[torch.randint(0, len(dis_cand), (pos.shape[1] * 12,), generator=g_rng)],
            ]
        )
        bad = set(pos_eval) | t_pairs
        keep = torch.tensor([(int(a), int(b)) not in bad for a, b in cand.T])
        negs = cand[:, keep][:, : pos.shape[1] * 10]
        sp, sn = score(h, pos).numpy(), score(h, negs).numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    s = np.r_[sp, sn]
    # hits@10 style: rank each positive among 10 sampled negatives of the same gene-block
    rank_ok = np.mean([(sp[i] > sn[i * 10 : (i + 1) * 10]).sum() >= 9 for i in range(len(sp))])
    return {
        "arm": arm,
        "seed": seed,
        "n_removed": int(remove.sum()),
        "n_train_pairs": int(len(uniq)),
        "n_eval_pos": int(len(sp)),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "hits_at_1_of_11": float(
            np.mean([(sp[i] > sn[i * 10 : (i + 1) * 10]).all() for i in range(len(sp))])
        ),
        "top2_of_11": float(rank_ok),
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--arms", default="original,gnn_flag,gbm_prosp,random,oracle,downweight")
    ap.add_argument("--tag", default="rgcn", help="which GNN's conformal flags to use")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    seed_everything(0)
    data = torch.load(PROC_DIR / "kg_T.pt", weights_only=False)
    vtab = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet")
    xidx = build_xref_index(parse_obo(RAW_DIR / "mondo" / "mondo_2019-11-25.obo"))
    pos_eval, t_pairs = new_pairs_latest(data, xidx, vtab)
    print(f"new gene-disease P/LP pairs by 2026-11: {len(pos_eval):,} (T pairs {len(t_pairs):,})")
    out = RESULTS_DIR / f"m6{args.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        masks = cleaning_masks(data, vtab, seed, args.tag, tuple(args.arms.split(",")))
        for arm in args.arms.split(","):
            r = train_eval(data, arm, masks[arm], pos_eval, t_pairs, seed, args.epochs)
            print(r, flush=True)
            rows.append(r)
            pd.DataFrame(rows).to_csv(out / "cleaning_runs.csv", index=False)
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("arm")[["auroc", "auprc", "hits_at_1_of_11", "n_removed"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(out / "cleaning_summary.csv")
    base = df[df["arm"] == "original"].set_index("seed")
    deltas = []
    for arm in df["arm"].unique():
        if arm == "original":
            continue
        a = df[df["arm"] == arm].set_index("seed")
        for m in ("auroc", "auprc", "hits_at_1_of_11"):
            d = (a[m] - base[m]).dropna()
            deltas.append(
                {
                    "arm": arm,
                    "metric": m,
                    "mean_delta_vs_original": float(d.mean()),
                    "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                    "n_seeds": int(len(d)),
                    "all_seeds_positive": bool((d > 0).all()),
                }
            )
    pd.DataFrame(deltas).to_csv(out / "cleaning_deltas.csv", index=False)
    write_json(
        {
            "n_eval_pos": len(pos_eval),
            "n_T_pairs": len(t_pairs),
            "seeds": args.seeds,
            "epochs": args.epochs,
        },
        out / "run.json",
    )
    print(agg.to_string())
    print(pd.DataFrame(deltas).to_string())


if __name__ == "__main__":
    main()
