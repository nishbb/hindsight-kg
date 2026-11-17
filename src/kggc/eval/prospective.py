"""Prospective training table: revisions observed before T (2018-01 versus 2020-01)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kggc.data.annotate import fetch_annotations
from kggc.data.clinvar import consequence_from_name, load_variant_summary
from kggc.eval.features import CONSEQ_ORDER, tabular_features
from kggc.utils import RAW_DIR

PROSPECTIVE_COLS = [
    "stars",
    "n_submitters",
    "no_assertion_criteria",
    "is_likely_pathogenic",
    "gnomad_af",
    "in_gnomad",
    "log10_gnomad_af",
] + [f"cons_{c}" for c in CONSEQ_ORDER]


def prospective_training_table() -> pd.DataFrame | None:
    """Revisions observed before T: class at 2018-01 versus class at 2020-01."""
    cv = RAW_DIR / "clinvar"
    p18, p20 = cv / "variant_summary_2018-01.txt.gz", cv / "variant_summary_2020-01.txt.gz"
    if not (p18.exists() and p20.exists()):
        return None
    t = load_variant_summary(p20)
    a2v = dict(zip(t["allele_id"], t["variation_id"].astype(int)))
    s = load_variant_summary(p18, allele_to_variation=a2v)
    s = s[
        (s["origin_simple"] == "germline")
        & s["clnsig_class"].isin(["PLP", "VUS"])
        & (s["gene_id"] > 0)
    ]
    s = s.merge(
        t[["variation_id", "clnsig_class"]].rename(columns={"clnsig_class": "cls_T"}),
        on="variation_id",
    )
    lab = pd.Series(pd.NA, index=s.index, dtype="object")
    a, b = s["clnsig_class"], s["cls_T"]
    lab[(a == "PLP") & (b == "PLP")] = "stable_pathogenic"
    lab[(a == "PLP") & b.isin(["VUS", "BLB", "CONFLICT"])] = "downgraded"
    lab[(a == "VUS") & (b == "PLP")] = "upgraded"
    lab[(a == "VUS") & (b == "VUS")] = "stable_vus"
    s["label"] = lab
    s = s[s["label"].notna()].copy()
    s["consequence"] = [consequence_from_name(n, ty) for n, ty in zip(s["name"], s["type"])]
    s["t_subclass"] = s["clnsig"]
    s["last_eval_year"] = np.nan
    ann = fetch_annotations(s["variation_id"].tolist())
    return tabular_features(s, ann)
