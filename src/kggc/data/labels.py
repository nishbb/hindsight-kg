"""Temporal ClinVar label table (M1).

A record enters the label table if, at snapshot T, it is
  * germline (OriginSimple == "germline", exact),
  * classified exactly P/LP or VUS (no compound terms),
  * linked to exactly one informative condition (after removing "not provided" etc.),
  * that condition maps to exactly one MONDO term via MONDO equivalentTo xrefs at T,
  * located in exactly one gene (GeneID > 0).
The label is then assigned from the aggregate classification of the same VariationID in the
latest release. Variants absent from the latest release, or whose latest class is outside
{PLP, VUS, BLB, CONFLICT}, are excluded and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kggc.data.clinvar import informative_conditions, split_conditions
from kggc.data.obo import normalize_curie

LABELS = ("downgraded", "stable_pathogenic", "upgraded", "stable_vus")

# Priority used when a condition carries several xrefs that map to different MONDO terms.
XREF_PRIORITY = ("MONDO", "OMIM", "Orphanet", "UMLS", "MESH", "HP")


def map_condition(xrefs: list[str], xref_index: dict[str, str]) -> str | None:
    """Return a single MONDO id for one condition, or None if unmapped/ambiguous."""
    by_prefix: dict[str, set[str]] = {}
    for x in xrefs:
        c = normalize_curie(x)
        m = xref_index.get(c)
        if m:
            by_prefix.setdefault(c.split(":")[0], set()).add(m)
    for pre in XREF_PRIORITY:
        hits = by_prefix.get(pre)
        if hits:
            return next(iter(hits)) if len(hits) == 1 else None
    return None


@dataclass
class LabelBuildReport:
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, name: str, n: int) -> None:
        self.steps.append((name, int(n)))


def resolve_single_condition(df: pd.DataFrame, xref_index: dict[str, str]) -> pd.DataFrame:
    """Add columns n_conditions, condition_name, condition_xrefs, mondo_id."""
    names, xrefs, n_conds, mondo = [], [], [], []
    for ids, lst in zip(df["phenotype_ids"].values, df["phenotype_list"].values):
        conds = informative_conditions(split_conditions(ids, lst))
        # Deduplicate identical conditions listed under several RCVs.
        uniq = {}
        for nm, xr in conds:
            uniq.setdefault(nm.lower(), (nm, xr))
        conds = list(uniq.values())
        n_conds.append(len(conds))
        if len(conds) == 1:
            nm, xr = conds[0]
            names.append(nm)
            xrefs.append(",".join(xr))
            mondo.append(map_condition(xr, xref_index))
        else:
            names.append("")
            xrefs.append("")
            mondo.append(None)
    out = df.copy()
    out["n_conditions"] = n_conds
    out["condition_name"] = names
    out["condition_xrefs"] = xrefs
    out["mondo_id"] = mondo
    return out


def build_labels(
    t_df: pd.DataFrame, latest_df: pd.DataFrame, xref_index: dict[str, str]
) -> tuple[pd.DataFrame, LabelBuildReport]:
    rep = LabelBuildReport()
    d = t_df
    rep.add("records_at_T", len(d))
    d = d[d["origin_simple"] == "germline"]
    rep.add("germline", len(d))
    d = d[d["clnsig_class"].isin(["PLP", "VUS"])]
    rep.add("exact_PLP_or_VUS", len(d))
    d = d[d["gene_id"] > 0]
    rep.add("single_gene", len(d))
    d = resolve_single_condition(d, xref_index)
    d = d[d["n_conditions"] == 1]
    rep.add("single_informative_condition", len(d))
    d = d[d["mondo_id"].notna()]
    rep.add("condition_maps_to_MONDO_at_T", len(d))

    lat = latest_df.set_index("variation_id")[["clnsig", "clnsig_class", "review_status", "stars"]]
    lat = lat.add_prefix("latest_")
    d = d.join(lat, on="variation_id", how="left")
    in_latest = d["latest_clnsig_class"].notna()
    rep.add("present_in_latest", int(in_latest.sum()))
    d = d[in_latest].copy()

    t_cls, l_cls = d["clnsig_class"], d["latest_clnsig_class"]
    label = pd.Series(pd.NA, index=d.index, dtype="object")
    label[(t_cls == "PLP") & (l_cls == "PLP")] = "stable_pathogenic"
    label[(t_cls == "PLP") & l_cls.isin(["VUS", "BLB", "CONFLICT"])] = "downgraded"
    label[(t_cls == "VUS") & (l_cls == "PLP")] = "upgraded"
    label[(t_cls == "VUS") & (l_cls == "VUS")] = "stable_vus"
    d["label"] = label
    # Kept for reporting only; not one of the four classes.
    d["vus_to_benign"] = (t_cls == "VUS") & (l_cls == "BLB")
    d["latest_outside_scope"] = label.isna() & ~d["vus_to_benign"]
    rep.add("latest_class_in_scope", int(label.notna().sum()))
    for lab in LABELS:
        rep.add(f"label_{lab}", int((label == lab).sum()))
    rep.add("vus_to_benign_not_labeled", int(d["vus_to_benign"].sum()))
    return d, rep


def assert_disjoint(labels: pd.DataFrame) -> None:
    lab = labels[labels["label"].notna()]
    counts = lab.groupby("variation_id")["label"].nunique()
    bad = counts[counts > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} variants appear in more than one class")
    if lab["variation_id"].duplicated().any():
        raise AssertionError("duplicate variation_id rows in label table")
