"""M3 (part 2): baselines, conformal calibration and the H1 evaluation.

Usage: python scripts/m3_evaluate.py [--n-boot 1000] [--tags rgcn hgt]
Reads the out-of-fold GNN scores written by scripts/m3_train_gnn.py.
Outputs (results/m3/): h1_main.csv, h1_deltas.csv, h1_by_stars.csv, h1_by_gene_group.csv,
  conformal_<tag>.json, feature_importance.csv
and data/processed/edge_scores.parquet (per labelled edge: features, scores, conformal sets).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from kggc.data.annotate import fetch_annotations
from kggc.eval.baselines import (
    SINGLE_DIRECTIONS,
    TASKS,
    gbm_cv_oof,
    gene_folds,
    make_gbm,
    single_feature_score,
    task_frame,
)
from kggc.eval.features import (
    ALLOWED_TIME_T,
    CADD,
    NEIGHBOR,
    TABULAR,
    attach_gnn,
    neighbor_counts,
    tabular_features,
)
from kggc.eval.metrics import auprc, auroc, bootstrap_ci, paired_bootstrap_delta
from kggc.eval.prospective import PROSPECTIVE_COLS, prospective_training_table
from kggc.graph.leakage import check_feature_columns
from kggc.models.conformal import calibrate_by_fold, coverage_report
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, seed_everything, write_json


def row(task, method, family, y, s, n_boot, seed=0):
    a, alo, ahi = bootstrap_ci(y, s, auroc, n_boot, seed)
    p, plo, phi = bootstrap_ci(y, s, auprc, n_boot, seed)
    return {
        "task": task,
        "method": method,
        "family": family,
        "n": len(y),
        "n_pos": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "auroc": a,
        "auroc_lo": alo,
        "auroc_hi": ahi,
        "auprc": p,
        "auprc_lo": plo,
        "auprc_hi": phi,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tags", nargs="*", default=["rgcn", "hgt"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--skip-prospective", action="store_true")
    ap.add_argument(
        "--only-scored",
        action="store_true",
        help="evaluate only edges that have a GNN score (subsampled runs)",
    )
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    seed_everything(0)
    out = RESULTS_DIR / f"m3{args.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(PROC_DIR / "labels.parquet")
    labels = labels[labels["label"].notna()].reset_index(drop=True)
    vtab = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet")
    ann_path = PROC_DIR / "annotations_T.parquet"
    if ann_path.exists():
        ann = pd.read_parquet(ann_path)
    else:
        ann = fetch_annotations(labels["variation_id"].tolist())
        ann.to_parquet(ann_path, index=False)
    feat = tabular_features(labels, ann)
    feat = feat.merge(neighbor_counts(vtab, labels), on="variation_id", how="left")

    tags = []
    for tag in args.tags:
        sp = PROC_DIR / f"gnn_scores_{tag}.parquet"
        if not sp.exists():
            print(f"[skip] no scores for {tag}")
            continue
        sc = pd.read_parquet(sp)
        cal = pd.read_parquet(PROC_DIR / f"gnn_calibration_{tag}.parquet")
        sc = calibrate_by_fold(sc, cal, alpha=args.alpha)
        cov = coverage_report(sc)
        cov["alpha"] = args.alpha
        cov["set_counts_plp_edges"] = (
            sc.loc[sc.edge_type == "plp", "conformal_set"].value_counts().to_dict()
        )
        cov["set_counts_vus_edges"] = (
            sc.loc[sc.edge_type == "vus", "conformal_set"].value_counts().to_dict()
        )
        write_json(cov, out / f"conformal_{tag}.json")
        print(tag, "conformal", cov)
        feat = attach_gnn(feat, sc, tag)
        extra = sc[["variation_id", "mondo_id", "conformal_set", "p_blb"]].drop_duplicates(
            ["variation_id", "mondo_id"]
        )
        feat = feat.merge(
            extra.rename(columns={"conformal_set": f"gnn_{tag}_set", "p_blb": f"gnn_{tag}_p_blb"}),
            on=["variation_id", "mondo_id"],
            how="left",
        )
        miss = feat[f"gnn_{tag}"].isna().mean()
        print(f"{tag}: labelled edges without a GNN score: {miss:.4%}")
        tags.append(tag)
    gnn_cols = [f"gnn_{t}" for t in tags]

    feature_sets = {
        "GBM tabular": TABULAR,
        "GBM tabular + CADD (version unverified)": TABULAR + CADD,
        "GBM tabular + neighbour counts": TABULAR + NEIGHBOR,
    }
    for t in tags:
        feature_sets[f"GBM tabular + GNN ({t})"] = TABULAR + [f"gnn_{t}"]
        feature_sets[f"GBM tabular + neighbour counts + GNN ({t})"] = (
            TABULAR + NEIGHBOR + [f"gnn_{t}"]
        )
    if len(tags) > 1:
        feature_sets["GBM tabular + GNN (all)"] = TABULAR + gnn_cols
        feature_sets["GBM tabular + neighbour counts + GNN (all)"] = TABULAR + NEIGHBOR + gnn_cols
    for name, cols in feature_sets.items():
        if "CADD" not in name:
            check_feature_columns(cols, ALLOWED_TIME_T)  # leakage guard

    prosp = None if args.skip_prospective else prospective_training_table()
    rows, deltas, strata_rows, group_rows, imp_rows = [], [], [], [], []
    hg = pd.read_csv(
        RAW_DIR / "hgnc" / "hgnc_complete_set.txt", sep="\t", dtype=str, low_memory=False
    )
    hg = hg[hg["entrez_id"].notna() & hg["gene_group"].notna()]
    gene_group = dict(zip(hg["entrez_id"].astype(int), hg["gene_group"].str.split("|").str[0]))

    if args.only_scored and gnn_cols:
        feat = feat[feat[gnn_cols].notna().all(axis=1)].reset_index(drop=True)
    for task in TASKS:
        d = task_frame(feat, task)
        for c in gnn_cols:
            d[c] = d[c].fillna(d[c].median())
        y = d["y"].to_numpy()
        fold = gene_folds(d["gene_id"].to_numpy(), 5, seed=0)
        preds: dict[str, np.ndarray] = {}
        for col in SINGLE_DIRECTIONS[task]:
            if col not in d:
                continue
            fam = "GNN (no outcome labels)" if col.startswith("gnn_") else "single feature"
            if col == "cadd_phred":
                fam = "single feature (version unverified)"
            preds[col] = single_feature_score(d, col, task)
            rows.append(row(task, col, fam, y, preds[col], args.n_boot))
        for name, cols in feature_sets.items():
            preds[name] = gbm_cv_oof(d, cols, fold)
            rows.append(
                row(task, name, "GBM, gene-grouped CV on outcomes", y, preds[name], args.n_boot)
            )
            print(task, name, f"AUROC {rows[-1]['auroc']:.4f} AUPRC {rows[-1]['auprc']:.4f}")
        if prosp is not None:
            ptr = task_frame(prosp, task)
            # exclude nothing: all training rows predate T by construction
            m = make_gbm(0).fit(ptr[PROSPECTIVE_COLS].to_numpy(float), ptr["y"].to_numpy())
            preds["GBM prospective (fit on 2018 to 2020 revisions)"] = m.predict_proba(
                d[PROSPECTIVE_COLS].to_numpy(float)
            )[:, 1]
            r = row(
                task,
                "GBM prospective (fit on 2018 to 2020 revisions)",
                "GBM, deployable at T",
                y,
                preds["GBM prospective (fit on 2018 to 2020 revisions)"],
                args.n_boot,
            )
            r["n_train"], r["n_train_pos"] = len(ptr), int(ptr["y"].sum())
            rows.append(r)
            print(
                task,
                "prospective",
                f"AUROC {r['auroc']:.4f} (train n={len(ptr)}, pos={r['n_train_pos']})",
            )

        pairs = []
        for t in tags:
            pairs.append(("GBM tabular", f"GBM tabular + GNN ({t})"))
            pairs.append(
                ("GBM tabular + neighbour counts", f"GBM tabular + neighbour counts + GNN ({t})")
            )
        pairs.append(("GBM tabular", "GBM tabular + neighbour counts"))
        if len(tags) > 1:
            pairs.append(("GBM tabular", "GBM tabular + GNN (all)"))
            pairs.append(
                ("GBM tabular + neighbour counts", "GBM tabular + neighbour counts + GNN (all)")
            )
        for a, b in pairs:
            for mname, fn in (("auroc", auroc), ("auprc", auprc)):
                dl, lo, hi, p = paired_bootstrap_delta(y, preds[a], preds[b], fn, args.n_boot)
                deltas.append(
                    {
                        "task": task,
                        "baseline": a,
                        "with_graph": b,
                        "metric": mname,
                        "delta": dl,
                        "lo": lo,
                        "hi": hi,
                        "p_boot": p,
                    }
                )

        # stratified results
        key_methods = ["GBM tabular"] + gnn_cols + [f"GBM tabular + GNN ({t})" for t in tags]
        star_bin = (
            d["stars"].clip(upper=3).map({0: "0 stars", 1: "1 star", 2: "2 stars", 3: "3+ stars"})
        )
        for sb in sorted(star_bin.unique()):
            m = (star_bin == sb).to_numpy()
            if y[m].sum() < 10 or (1 - y[m]).sum() < 10:
                continue
            for meth in key_methods:
                r = row(task, meth, "", y[m], preds[meth][m], max(200, args.n_boot // 5))
                r["stratum"] = sb
                strata_rows.append(r)
        grp = d["gene_id"].map(gene_group).fillna("(no HGNC group)")
        top = d.assign(grp=grp).groupby("grp")["y"].agg(["sum", "size"])
        top = (
            top[(top["sum"] >= 20) & (top.index != "(no HGNC group)")]
            .sort_values("sum", ascending=False)
            .head(12)
        )
        for gname in top.index:
            m = (grp == gname).to_numpy()
            if (1 - y[m]).sum() < 10:
                continue
            for meth in key_methods:
                r = row(task, meth, "", y[m], preds[meth][m], max(200, args.n_boot // 5))
                r["stratum"] = gname
                group_rows.append(r)

        # permutation importance of the richest model (fit on 4 folds, scored on the 5th)
        cols = TABULAR + NEIGHBOR + gnn_cols
        tr, te = fold != 0, fold == 0
        mdl = make_gbm(0).fit(d.loc[tr, cols].to_numpy(float), y[tr])
        pi = permutation_importance(
            mdl,
            d.loc[te, cols].to_numpy(float),
            y[te],
            scoring="roc_auc",
            n_repeats=5,
            random_state=0,
        )
        for c, mu, sd in zip(cols, pi.importances_mean, pi.importances_std):
            imp_rows.append({"task": task, "feature": c, "auroc_drop": mu, "sd": sd})

        d_out = d[["variation_id", "y"]].copy()
        for name in ("GBM tabular", *[f"GBM tabular + GNN ({t})" for t in tags]):
            feat.loc[feat["variation_id"].isin(d["variation_id"]), f"oof::{name}"] = (
                pd.Series(preds[name], index=d["variation_id"])
                .reindex(feat.loc[feat["variation_id"].isin(d["variation_id"]), "variation_id"])
                .values
            )
        del d_out

    pd.DataFrame(rows).to_csv(out / "h1_main.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "h1_deltas.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(out / "h1_by_stars.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "h1_by_gene_group.csv", index=False)
    pd.DataFrame(imp_rows).sort_values(["task", "auroc_drop"], ascending=[True, False]).to_csv(
        out / "feature_importance.csv", index=False
    )
    feat.to_parquet(PROC_DIR / f"edge_scores{args.out_suffix}.parquet", index=False)
    print(
        pd.DataFrame(rows)[["task", "method", "auroc", "auroc_lo", "auroc_hi", "auprc"]].to_string()
    )
    print(pd.DataFrame(deltas).to_string())


if __name__ == "__main__":
    main()
