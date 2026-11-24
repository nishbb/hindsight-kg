"""M5: paired bootstrap comparisons between audit configurations (same 320 edges).

Usage: python scripts/m5_stats.py [--n-boot 2000]
Output: results/m5/paired_deltas.csv and results/m5/trivial_baseline.csv (always answer "keep")
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from kggc.utils import RESULTS_DIR

PAIRS = [
    ("keep", "a"),
    ("a", "b0"),
    ("a", "b"),
    ("a", "c"),
    ("a", "d"),
    ("b0", "b"),
    ("b", "c"),
    ("c", "d"),
    ("b", "d"),
]
LABELS = ["keep", "downgrade", "upgrade"]


def metrics(y, p):
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = RESULTS_DIR / "m5"
    v = {}
    for cfg in ("a", "b0", "b", "c", "d"):
        p = out / f"verdicts_{cfg}.csv"
        if p.exists():
            d = pd.read_csv(p).sort_values("variation_id").reset_index(drop=True)
            v[cfg] = d
    base = next(iter(v.values()))
    y = base["truth"].to_numpy()
    v["keep"] = base.assign(verdict="keep")  # trivial baseline: never revise anything
    pd.DataFrame([{"config": "keep", **metrics(y, v["keep"]["verdict"].to_numpy())}]).to_csv(
        out / "trivial_baseline.csv", index=False
    )
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.n_boot)]
    rows = []
    for a, b in PAIRS:
        if a not in v or b not in v:
            continue
        assert (v[a]["variation_id"].to_numpy() == v[b]["variation_id"].to_numpy()).all()
        pa, pb = v[a]["verdict"].to_numpy(), v[b]["verdict"].to_numpy()
        for m in ("accuracy", "macro_f1"):
            point = metrics(y, pb)[m] - metrics(y, pa)[m]
            boots = np.array([metrics(y[i], pb[i])[m] - metrics(y[i], pa[i])[m] for i in idx])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            rows.append(
                {
                    "from": a,
                    "to": b,
                    "metric": m,
                    "delta": point,
                    "lo": lo,
                    "hi": hi,
                    "p_boot": min(1.0, p),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "paired_deltas.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
