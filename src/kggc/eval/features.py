"""Time-T tabular features, neighbourhood counts and the feature allow list."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSEQ_ORDER = [
    "missense",
    "nonsense",
    "frameshift",
    "splice_site",
    "inframe_indel",
    "synonymous",
    "start_lost",
    "intronic",
    "utr5",
    "utr3",
    "structural_or_other",
    "other",
]

# ClinVar metadata + gnomAD v2.1.1: everything a curator could see at T.
TABULAR = [
    "stars",
    "n_submitters",
    "no_assertion_criteria",
    "ever_conflicting_before_T",
    "class_changed_before_T",
    "ever_vus_before_T",
    "ever_blb_before_T",
    "n_prior_snapshots_present",
    "years_since_last_eval",
    "is_likely_pathogenic",
    "gnomad_af",
    "in_gnomad",
    "log10_gnomad_af",
] + [f"cons_{c}" for c in CONSEQ_ORDER]

# Reported separately: MyVariant.info does not state the CADD release, so we cannot verify
# that it predates T. CADD is not trained on ClinVar labels.
CADD = ["cadd_phred", "has_cadd"]

# Simple graph statistics (leave-one-out over the other variants in the time-T graph).
NEIGHBOR = [
    "gene_n_plp",
    "gene_n_vus",
    "gene_n_blb",
    "gene_n_conflict",
    "gene_frac_plp",
    "gene_disease_n_plp",
    "gene_same_cons_n_plp",
    "gene_same_cons_n_blb",
    "disease_n_plp",
]

GNN = ["gnn_rgcn", "gnn_hgt"]

ALLOWED_TIME_T = set(TABULAR) | set(NEIGHBOR) | set(GNN) | {"gnn_rgcn_p_plp", "gnn_hgt_p_plp"}


def tabular_features(labels: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    d = labels.merge(
        ann[["variation_id", "gnomad_af", "in_gnomad", "cadd_phred"]], on="variation_id", how="left"
    )
    d["gnomad_af"] = d["gnomad_af"].fillna(0.0)
    d["in_gnomad"] = d["in_gnomad"].fillna(0).astype(int)
    d["log10_gnomad_af"] = np.log10(d["gnomad_af"].clip(lower=1e-7))
    d["has_cadd"] = d["cadd_phred"].notna().astype(int)
    d["no_assertion_criteria"] = (d["review_status"] == "no assertion criteria provided").astype(
        int
    )
    d["is_likely_pathogenic"] = (d["t_subclass"] == "Likely pathogenic").astype(int)
    d["years_since_last_eval"] = (2020 - d["last_eval_year"]).clip(lower=0)
    for c in CONSEQ_ORDER:
        d[f"cons_{c}"] = (d["consequence"] == c).astype(int)
    return d


def neighbor_counts(vtab: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out counts of other time-T ClinVar variants in the same gene / disease.

    ``vtab`` is the time-T variant node table (ClinVar variants, any number of conditions).
    Disease-specific counts use the single mapped disease of each labelled variant and the
    single-condition subset of the graph variants (the label table itself).
    """
    cv = vtab[vtab["origin"] == "clinvar"]
    g = cv.groupby(["gene_id", "clnsig_class_T"]).size().unstack(fill_value=0)
    for c in ("PLP", "VUS", "BLB", "CONFLICT"):
        if c not in g:
            g[c] = 0
    gc = cv.groupby(["gene_id", "consequence", "clnsig_class_T"]).size().unstack(fill_value=0)
    for c in ("PLP", "BLB"):
        if c not in gc:
            gc[c] = 0

    out = labels[["variation_id", "gene_id", "mondo_id", "clnsig_class", "consequence"]].copy()
    own = out["clnsig_class"]
    base = g.reindex(out["gene_id"]).fillna(0)
    out["gene_n_plp"] = base["PLP"].values - (own == "PLP").values
    out["gene_n_vus"] = base["VUS"].values - (own == "VUS").values
    out["gene_n_blb"] = base["BLB"].values
    out["gene_n_conflict"] = base["CONFLICT"].values
    tot = out[["gene_n_plp", "gene_n_vus", "gene_n_blb", "gene_n_conflict"]].sum(axis=1)
    out["gene_frac_plp"] = np.where(tot > 0, out["gene_n_plp"] / tot.clip(lower=1), 0.5)
    key = pd.MultiIndex.from_arrays([out["gene_id"], out["consequence"]])
    sc = gc.reindex(key).fillna(0)
    out["gene_same_cons_n_plp"] = sc["PLP"].values - (own == "PLP").values
    out["gene_same_cons_n_blb"] = sc["BLB"].values

    plp = labels[labels["clnsig_class"] == "PLP"]
    gd = plp.groupby(["gene_id", "mondo_id"]).size()
    dd = plp.groupby("mondo_id").size()
    key = pd.MultiIndex.from_arrays([out["gene_id"], out["mondo_id"]])
    out["gene_disease_n_plp"] = gd.reindex(key).fillna(0).values - (own == "PLP").values
    out["disease_n_plp"] = dd.reindex(out["mondo_id"]).fillna(0).values - (own == "PLP").values
    return out.drop(columns=["gene_id", "mondo_id", "clnsig_class", "consequence"])


def attach_gnn(feat: pd.DataFrame, scores: pd.DataFrame, name: str) -> pd.DataFrame:
    """Join out-of-fold GNN scores on (variation_id, mondo_id)."""
    s = scores[["variation_id", "mondo_id", "score"] + [c for c in ("p_plp",) if c in scores]]
    s = s.drop_duplicates(["variation_id", "mondo_id"]).rename(
        columns={"score": f"gnn_{name}", "p_plp": f"gnn_{name}_p_plp"}
    )
    return feat.merge(s, on=["variation_id", "mondo_id"], how="left")
