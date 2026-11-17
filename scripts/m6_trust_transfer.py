"""M6 (part 2 preparation): transfer kg-gc trust onto PrimeKG / STaRK-Prime disease-protein edges.

PrimeKG has no variant nodes, so variant-level trust cannot be used directly. For each
(gene, MONDO disease) pair we aggregate the time-T ClinVar P/LP edges that support it:

  n_plp            P/LP variant edges at T supporting the pair
  n_flagged        of these, edges whose out-of-fold R-GCN conformal set is {blb}
  trust            1 - n_flagged / n_plp (1.0 when no supporting edge is flagged)

PrimeKG disease ids are MONDO numbers, or '_'-joined groups of MONDO numbers ('MONDO_grouped');
a PrimeKG edge gets the minimum trust over its member diseases. Edges with no ClinVar support
at T get trust NaN (unknown, not low).

Output: results/m6_stark/primekg_disease_protein_trust.csv.gz (+ coverage summary).
The G-Retriever training and evaluation on STaRK-Prime (original versus trust-pruned or
trust-weighted PCST) needs a GPU and has not been run; see NOTES.md.
"""

from __future__ import annotations

import pandas as pd

from kggc.models.conformal import calibrate_by_fold
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, write_json


def main() -> None:
    out = RESULTS_DIR / "m6_stark"
    out.mkdir(parents=True, exist_ok=True)
    sc = calibrate_by_fold(
        pd.read_parquet(PROC_DIR / "gnn_scores_rgcn.parquet"),
        pd.read_parquet(PROC_DIR / "gnn_calibration_rgcn.parquet"),
    )
    sc = sc[sc["edge_type"] == "plp"]
    vt = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet")[["variation_id", "gene_id"]]
    sc = sc.merge(vt, on="variation_id")
    sc = sc[sc["gene_id"] > 0]
    sc["flagged"] = sc["conformal_set"] == "blb"
    pair = (
        sc.groupby(["gene_id", "mondo_id"])
        .agg(n_plp=("flagged", "size"), n_flagged=("flagged", "sum"))
        .reset_index()
    )
    pair["trust"] = 1 - pair["n_flagged"] / pair["n_plp"]
    pair["mondo_num"] = (
        pair["mondo_id"].str.replace("MONDO:", "", regex=False).astype(int).astype(str)
    )

    kg = pd.read_csv(
        RAW_DIR / "primekg" / "kg.csv",
        usecols=["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name", "y_source"],
        dtype=str,
    )
    dp = kg[(kg["relation"] == "disease_protein") & (kg["x_type"] == "gene/protein")].copy()
    dp["members"] = dp["y_id"].str.split("_")
    ex = dp.explode("members")
    ex["members"] = ex["members"].astype(int).astype(str)
    ex = ex.merge(
        pair,
        left_on=["x_id", "members"],
        right_on=[pair["gene_id"].astype(str), "mondo_num"],
        how="left",
    )
    agg = (
        ex.groupby(["x_id", "y_id"])
        .agg(trust=("trust", "min"), n_plp=("n_plp", "sum"), n_flagged=("n_flagged", "sum"))
        .reset_index()
    )
    dp = dp.drop(columns="members").merge(agg, on=["x_id", "y_id"], how="left")
    dp[["x_id", "x_name", "y_id", "y_name", "y_source", "n_plp", "n_flagged", "trust"]].to_csv(
        out / "primekg_disease_protein_trust.csv.gz", index=False
    )
    cov = {
        "primekg_disease_protein_edges": int(len(dp)),
        "edges_with_clinvar_support_at_T": int(dp["trust"].notna().sum()),
        "edges_with_any_flagged_support": int((dp["n_flagged"] > 0).sum()),
        "edges_with_trust_below_0.5": int((dp["trust"] < 0.5).sum()),
        "note": "Trust covers only gene-disease edges with time-T ClinVar P/LP support; PrimeKG edges from other sources stay unscored.",
    }
    write_json(cov, out / "coverage.json")
    print(cov)


if __name__ == "__main__":
    main()
