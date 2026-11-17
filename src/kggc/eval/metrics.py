"""Metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(y, s, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate and percentile bootstrap CI, resampling within class (stratified)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = fn(y, s)
    if len(pos) == 0 or len(neg) == 0:
        return point, float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals[b] = fn(y[idx], s[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap_delta(y, s_a, s_b, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Delta fn(b) - fn(a) on the same resamples; returns (delta, lo, hi, p_two_sided)."""
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = fn(y, s_b) - fn(y, s_a)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals[b] = fn(y[idx], s_b[idx]) - fn(y[idx], s_a[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(point), float(lo), float(hi), float(min(1.0, p))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error for probabilities of the positive class."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def ece_confidence(correct, conf, n_bins: int = 10) -> float:
    """ECE for multi-class decisions: |accuracy - confidence| per confidence bin."""
    return ece(correct, conf, n_bins)
