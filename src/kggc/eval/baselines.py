"""Baselines for H1: single time-T features and gradient-boosted models.

Two supervised regimes are reported and must not be confused:

* ``gbm_cv``: gene-grouped K-fold cross-validation on the 2020 -> 2026 outcome labels. This
  measures how much signal a feature set carries. It is a retrospective analysis (nobody had
  these labels in 2020). Features are still strictly time-T.
* ``gbm_prospective``: a model fitted on revisions observed *before* T (2018-01 -> 2020-01)
  and applied unchanged to the 2020-01 records. This is what could actually have been
  deployed at T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

TASKS = {
    "downgrade": ("downgraded", "stable_pathogenic"),
    "upgrade": ("upgraded", "stable_vus"),
}

# A-priori direction of each single feature (+1: larger value means more likely positive).
SINGLE_DIRECTIONS = {
    "downgrade": {
        "stars": -1,
        "n_submitters": -1,
        "ever_conflicting_before_T": +1,
        "gnomad_af": +1,
        "cadd_phred": -1,
        "gnn_rgcn": -1,
        "gnn_hgt": -1,
    },
    "upgrade": {
        "stars": +1,
        "n_submitters": +1,
        "ever_conflicting_before_T": +1,
        "gnomad_af": -1,
        "cadd_phred": +1,
        "gnn_rgcn": +1,
        "gnn_hgt": +1,
    },
}


def task_frame(feat: pd.DataFrame, task: str) -> pd.DataFrame:
    pos, neg = TASKS[task]
    d = feat[feat["label"].isin([pos, neg])].copy()
    d["y"] = (d["label"] == pos).astype(int)
    return d.reset_index(drop=True)


def make_gbm(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def gene_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row; all variants of a gene share a fold (seeded)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    remap = dict(zip(uniq, rng.permutation(len(uniq))))
    g = np.array([remap[x] for x in groups])
    fold = np.empty(len(groups), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(g, groups=g)):
        fold[te] = k
    return fold


def gbm_cv_oof(d: pd.DataFrame, cols: list[str], fold: np.ndarray, seed: int = 0) -> np.ndarray:
    oof = np.full(len(d), np.nan)
    X, y = d[cols].to_numpy(dtype=float), d["y"].to_numpy()
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = make_gbm(seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def single_feature_score(d: pd.DataFrame, col: str, task: str) -> np.ndarray:
    s = d[col].astype(float)
    s = s.fillna(s.median())
    return (SINGLE_DIRECTIONS[task][col] * s).to_numpy()
