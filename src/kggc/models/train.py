"""Cross-fitted training and scoring of the edge-plausibility GNN (M3).

Every ClinVar variant is assigned to one of K folds. For fold k the model is trained without
any assertion edge of fold-k variants, then scores those edges. Each P/LP edge and each VUS
pair therefore gets an out-of-sample plausibility score, using time-T information only.

Within the training variants, 10% are a validation set (never in the message graph) used for
early stopping and as the split-conformal calibration set.

Training uses full-batch message passing. With ``pyg-lib`` or ``torch-sparse`` installed, the
neighbour-sampled path in ``kggc.models.sampled`` can be used instead (GPU instance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from kggc.models.gnn import (
    ASSERT_TYPES,
    HeteroLinkPredictor,
    all_edge_types,
    message_edge_dict,
)


@dataclass
class TrainConfig:
    encoder: str = "rgcn"
    hidden: int = 64
    layers: int = 2
    heads: int = 2
    dropout: float = 0.2
    lr: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    eval_every: int = 2
    patience: int = 4  # in evaluations
    sup_frac: float = 0.3
    folds: int = 5
    val_frac: float = 0.1
    seed: int = 0
    device: str = "cpu"
    drop_edge_types: list[str] = field(default_factory=list)  # relation names to drop
    max_genes: int | None = None  # subsample for the small reproduction


def static_edges(data, drop: list[str]) -> dict:
    out = {}
    for et in data.edge_types:
        s, r, d = et
        if r.startswith("assoc_") or r in drop:
            continue
        ei = data[et].edge_index
        out[et] = ei
        out[(d, f"rev_{r}", s)] = ei.flip(0)
    return out


def variant_gene_index(data) -> torch.Tensor:
    n = data["variant"].num_nodes
    vg = torch.full((n,), -1, dtype=torch.long)
    ei = data["variant", "in_gene", "gene"].edge_index
    # if a variant has several gene edges keep the first after sorting (deterministic)
    order = torch.argsort(ei[0] * (int(ei[1].max()) + 1) + ei[1], descending=True)
    vg[ei[0][order]] = ei[1][order]
    return vg


def assign_folds(vtab: pd.DataFrame, cfg: TrainConfig) -> np.ndarray:
    """fold id per variant node; -1 for GWAS-only variants (no assertion edges)."""
    rng = np.random.default_rng(cfg.seed)
    vt = vtab.sort_values("node")
    fold = np.full(len(vt), -1, dtype=np.int64)
    is_cv = (vt["origin"] == "clinvar").values
    if cfg.max_genes:
        genes = np.unique(vt.loc[is_cv, "gene_id"].values)
        pick = set(rng.choice(genes, size=min(cfg.max_genes, len(genes)), replace=False).tolist())
        is_cv = is_cv & vt["gene_id"].isin(pick).values
    idx = np.flatnonzero(is_cv)
    fold[idx] = rng.permutation(len(idx)) % cfg.folds
    return fold


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def run_fold(data, vtab, fold_ids: np.ndarray, k: int, cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed * 1000 + k)
    rng = np.random.default_rng(cfg.seed * 1000 + k)
    dev = torch.device(cfg.device)
    static = {et: ei.to(dev) for et, ei in static_edges(data, cfg.drop_edge_types).items()}
    vgene = variant_gene_index(data).to(dev)
    vx = data["variant"].x.to(dev)

    in_scope = fold_ids >= 0
    test_v = fold_ids == k
    train_pool = np.flatnonzero(in_scope & ~test_v)
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * cfg.val_frac)
    val_mask = np.zeros(len(fold_ids), dtype=bool)
    val_mask[train_pool[:n_val]] = True
    train_mask = np.zeros(len(fold_ids), dtype=bool)
    train_mask[train_pool[n_val:]] = True
    tm, vm, sm = (torch.from_numpy(m) for m in (train_mask, val_mask, test_v))

    edges = {t: data["variant", f"assoc_{t}", "disease"].edge_index for t in ASSERT_TYPES}
    tr = {t: e[:, tm[e[0]]].to(dev) for t, e in edges.items()}
    va = {t: e[:, vm[e[0]]].to(dev) for t, e in edges.items()}
    te = {t: e[:, sm[e[0]]].to(dev) for t, e in edges.items()}
    # candidate diseases for corrupted negatives: diseases with any time-T assertion
    cand = torch.unique(torch.cat([e[1] for e in edges.values()])).to(dev)

    num_nodes = {nt: int(data[nt].num_nodes) for nt in data.node_types}
    model = HeteroLinkPredictor(
        num_nodes,
        vx.shape[1],
        all_edge_types(list(static)),
        hidden=cfg.hidden,
        layers=cfg.layers,
        encoder=cfg.encoder,
        heads=cfg.heads,
        dropout=cfg.dropout,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def corrupt(pos: torch.Tensor) -> torch.Tensor:
        rd = cand[torch.randint(0, len(cand), (pos.shape[1],), device=dev)]
        return torch.stack([pos[0], rd])

    full_msg = message_edge_dict(static, tr, vgene)

    @torch.no_grad()
    def evaluate(split: dict) -> dict:
        model.eval()
        h = model.encode(vx, full_msg)
        s = {t: model.decode(h, split[t]).float().cpu().numpy() for t in ASSERT_TYPES}
        neg = model.decode(h, corrupt(split["plp"])).float().cpu().numpy()
        return {
            "scores": s,
            "auc_plp_vs_blb": _auc(s["plp"], s["blb"]),
            "auc_plp_vs_corrupt": _auc(s["plp"], neg),
        }

    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        msg, sup = {}, {}
        for t in ASSERT_TYPES:
            e = tr[t]
            if t in ("plp", "blb"):
                perm = torch.randperm(e.shape[1], device=dev)
                ns = int(cfg.sup_frac * e.shape[1])
                sup[t], msg[t] = e[:, perm[:ns]], e[:, perm[ns:]]
            else:
                msg[t] = e
        h = model.encode(vx, message_edge_dict(static, msg, vgene))
        pos = model.decode(h, sup["plp"])
        hard = model.decode(h, sup["blb"])
        rnd = model.decode(h, corrupt(sup["plp"]))
        loss = (
            F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + 0.5 * F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))
            + 0.5 * F.binary_cross_entropy_with_logits(rnd, torch.zeros_like(rnd))
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % cfg.eval_every == 0 or ep == cfg.epochs:
            ev = evaluate(va)
            metric = np.nanmean([ev["auc_plp_vs_blb"], ev["auc_plp_vs_corrupt"]])
            hist.append(
                {
                    "epoch": ep,
                    "loss": float(loss.detach()),
                    "val_auc_plp_vs_blb": ev["auc_plp_vs_blb"],
                    "val_auc_plp_vs_corrupt": ev["auc_plp_vs_corrupt"],
                }
            )
            log(
                f"[{cfg.encoder} fold {k}] ep {ep:3d} loss {float(loss.detach()):.4f} "
                f"val AUC blb {ev['auc_plp_vs_blb']:.4f} corrupt {ev['auc_plp_vs_corrupt']:.4f} "
                f"({time.time() - t0:.0f}s)"
            )
            if metric > best + 1e-4:
                best, bad = metric, 0
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    ev_val, ev_test = evaluate(va), evaluate(te)

    def frame(split: dict, ev: dict, name: str) -> pd.DataFrame:
        parts = []
        for t in ASSERT_TYPES:
            e = split[t].cpu().numpy()
            parts.append(
                pd.DataFrame(
                    {
                        "variant_node": e[0],
                        "disease_node": e[1],
                        "edge_type": t,
                        "score": ev["scores"][t],
                        "fold": k,
                        "split": name,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    return {
        "test": frame(te, ev_test, "test"),
        "val": frame(va, ev_val, "val"),
        "metrics": {
            "fold": k,
            "encoder": cfg.encoder,
            "epochs_run": hist[-1]["epoch"] if hist else 0,
            "val_auc_plp_vs_blb": ev_val["auc_plp_vs_blb"],
            "val_auc_plp_vs_corrupt": ev_val["auc_plp_vs_corrupt"],
            "test_auc_plp_vs_blb": ev_test["auc_plp_vs_blb"],
            "test_auc_plp_vs_corrupt": ev_test["auc_plp_vs_corrupt"],
            "seconds": time.time() - t0,
            "n_train_plp": int(tr["plp"].shape[1]),
            "n_val_plp": int(va["plp"].shape[1]),
            "n_test_plp": int(te["plp"].shape[1]),
        },
        "history": hist,
        "state": best_state,
    }


def cross_fit(data, vtab, cfg: TrainConfig, log=print) -> dict:
    fold_ids = assign_folds(vtab, cfg)
    tests, vals, mets, hists = [], [], [], []
    for k in range(cfg.folds):
        r = run_fold(data, vtab, fold_ids, k, cfg, log)
        tests.append(r["test"])
        vals.append(r["val"])
        mets.append(r["metrics"])
        hists += [dict(h, fold=k) for h in r["history"]]
    return {
        "scores": pd.concat(tests, ignore_index=True),
        "calibration": pd.concat(vals, ignore_index=True),
        "metrics": pd.DataFrame(mets),
        "history": pd.DataFrame(hists),
        "fold_ids": fold_ids,
    }
