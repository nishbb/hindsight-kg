"""Pre-T assertion-history features from earlier ClinVar snapshots (all dated before T)."""

from __future__ import annotations

import pandas as pd


def history_features(snapshots: list[pd.DataFrame], ids: pd.Series) -> pd.DataFrame:
    """For each VariationID: seen in each earlier snapshot, class there, conflicts, changes.

    ``snapshots`` must all predate T; the caller is responsible for that and the leakage test
    checks the snapshot dates recorded in the manifest.
    """
    out = pd.DataFrame({"variation_id": ids.values}).drop_duplicates().set_index("variation_id")
    classes = []
    for snap in snapshots:
        tag = snap["snapshot"].iloc[0]
        s = snap.set_index("variation_id")["clnsig_class"]
        col = f"cls_{tag}"
        out[col] = s.reindex(out.index)
        classes.append(col)
    present = out[classes].notna()
    out["n_prior_snapshots_present"] = present.sum(axis=1)
    out["ever_conflicting_before_T"] = (out[classes] == "CONFLICT").any(axis=1).astype(int)
    out["ever_vus_before_T"] = (out[classes] == "VUS").any(axis=1).astype(int)
    out["ever_blb_before_T"] = (out[classes] == "BLB").any(axis=1).astype(int)

    def _changed(row) -> int:
        vals = [v for v in row if isinstance(v, str)]
        return int(len(set(vals)) > 1)

    out["class_changed_before_T"] = out[classes].apply(_changed, axis=1)
    return out.reset_index()
