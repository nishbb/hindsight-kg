"""Split conformal calibration for edge-plausibility scores (class-conditional / Mondrian).

Calibration uses validation edges whose time-T class is known (P/LP vs B/LB) and which were
never in the message graph. For a new edge with logit ``s`` and p = sigmoid(s):

* nonconformity for class PLP is ``1 - p``; for class BLB it is ``p``;
* the conformal p-value for class c is ``(1 + #{cal_c >= a}) / (n_c + 1)``;
* the prediction set at miscoverage ``alpha`` is ``{c : p_value_c > alpha}``.

Under exchangeability each class is covered with probability >= 1 - alpha. The guarantee is
about the *time-T* label (does this edge look like a P/LP assertion or a B/LB assertion), not
about future reclassification; whether low conformal support predicts later downgrade is the
empirical H1 question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class SplitConformal:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.cal_plp: np.ndarray | None = None
        self.cal_blb: np.ndarray | None = None

    def fit(self, scores_plp: np.ndarray, scores_blb: np.ndarray) -> SplitConformal:
        self.cal_plp = np.sort(1.0 - _sigmoid(scores_plp))
        self.cal_blb = np.sort(_sigmoid(scores_blb))
        return self

    @staticmethod
    def _pvalue(cal_sorted: np.ndarray, a: np.ndarray) -> np.ndarray:
        n = len(cal_sorted)
        ge = n - np.searchsorted(cal_sorted, a, side="left")
        return (1.0 + ge) / (n + 1.0)

    def p_values(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _sigmoid(scores)
        return self._pvalue(self.cal_plp, 1.0 - p), self._pvalue(self.cal_blb, p)

    def predict(self, scores: np.ndarray) -> pd.DataFrame:
        p_plp, p_blb = self.p_values(scores)
        in_plp, in_blb = p_plp > self.alpha, p_blb > self.alpha
        set_label = np.select(
            [in_plp & ~in_blb, ~in_plp & in_blb, in_plp & in_blb],
            ["plp", "blb", "both"],
            default="empty",
        )
        return pd.DataFrame(
            {
                "p_plp": p_plp,
                "p_blb": p_blb,
                "conformal_set": set_label,
                # 1 when the set is a single class, 0 when ambiguous or empty
                "conformal_confident": np.isin(set_label, ["plp", "blb"]).astype(int),
            }
        )


def calibrate_by_fold(
    scores: pd.DataFrame, calibration: pd.DataFrame, alpha: float = 0.1
) -> pd.DataFrame:
    """Add conformal columns to the out-of-fold score table, calibrating each fold separately."""
    parts = []
    for k, grp in scores.groupby("fold"):
        cal = calibration[calibration["fold"] == k]
        sc = SplitConformal(alpha).fit(
            cal.loc[cal["edge_type"] == "plp", "score"].values,
            cal.loc[cal["edge_type"] == "blb", "score"].values,
        )
        parts.append(
            pd.concat([grp.reset_index(drop=True), sc.predict(grp["score"].values)], axis=1)
        )
    return pd.concat(parts, ignore_index=True)


def coverage_report(df: pd.DataFrame) -> dict:
    """Empirical coverage of the true time-T class on held-out P/LP and B/LB edges."""
    out = {}
    for cls in ("plp", "blb"):
        sub = df[df["edge_type"] == cls]
        if len(sub) == 0:
            continue
        covered = sub["conformal_set"].isin([cls, "both"]).mean()
        out[f"coverage_{cls}"] = float(covered)
        out[f"n_{cls}"] = int(len(sub))
        out[f"singleton_rate_{cls}"] = float(sub["conformal_set"].isin(["plp", "blb"]).mean())
    return out
