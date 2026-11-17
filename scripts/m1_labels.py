"""M1: temporal ClinVar labels.

Usage: python scripts/m1_labels.py [--small]
Outputs:
  data/processed/clinvar_T.parquet        all GRCh38-preferred records at T (for M2)
  data/processed/labels.parquet           label table with time-T features
  results/m1/*.csv, results/m1/summary.json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from kggc.data.clinvar import consequence_from_name, load_variant_summary
from kggc.data.history import history_features
from kggc.data.labels import LABELS, assert_disjoint, build_labels
from kggc.data.obo import build_xref_index, parse_obo
from kggc.data.sources import manifest
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, seed_everything, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    out = RESULTS_DIR / "m1"
    out.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    cv = RAW_DIR / "clinvar"
    print("loading ClinVar snapshots ...")
    t_df = load_variant_summary(cv / "variant_summary_2020-01.txt.gz")
    latest = load_variant_summary(cv / "variant_summary_2026-11.txt.gz")
    a2v = dict(zip(t_df["allele_id"], t_df["variation_id"].astype(int)))
    s18 = load_variant_summary(cv / "variant_summary_2018-01.txt.gz", allele_to_variation=a2v)
    s19 = load_variant_summary(cv / "variant_summary_2019-01.txt.gz")
    print(f"T={len(t_df):,} latest={len(latest):,} 2018={len(s18):,} 2019={len(s19):,}")

    mondo = parse_obo(RAW_DIR / "mondo" / "mondo_2019-11-25.obo")
    xidx = build_xref_index(mondo, equivalent_only=True)
    print(f"MONDO terms={len(mondo):,} xref index={len(xidx):,}")

    labels, rep = build_labels(t_df, latest, xidx)
    assert_disjoint(labels)

    hist = history_features([s18, s19, t_df], labels["variation_id"])
    labels = labels.merge(hist, on="variation_id", how="left")
    labels["consequence"] = [
        consequence_from_name(n, t) for n, t in zip(labels["name"], labels["type"])
    ]
    labels["t_subclass"] = labels["clnsig"]  # e.g. Pathogenic vs Likely pathogenic at T
    labels["last_eval_year"] = pd.to_datetime(
        labels["last_evaluated"], errors="coerce", format="mixed"
    ).dt.year
    # Leakage guard: a LastEvaluated date after T would mean the T record is not what we think.
    late = pd.to_datetime(labels["last_evaluated"], errors="coerce", format="mixed") > "2020-01-02"
    rep.add("rows_with_LastEvaluated_after_T_release(dropped)", int(late.sum()))
    labels = labels[~late]

    keep_cols = [c for c in labels.columns if not c.startswith("cls_")] + [
        c for c in labels.columns if c.startswith("cls_")
    ]
    labels[keep_cols].to_parquet(PROC_DIR / "labels.parquet", index=False)
    t_df.to_parquet(PROC_DIR / "clinvar_T.parquet", index=False)
    write_json(manifest(), PROC_DIR / "source_manifest.json")

    lab = labels[labels["label"].notna()]
    counts = lab["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    counts.to_csv(out / "label_counts.csv", header=["n"])
    pd.DataFrame(rep.steps, columns=["step", "n"]).to_csv(out / "filter_steps.csv", index=False)

    by_review = pd.crosstab(lab["review_status"], lab["label"]).reindex(
        columns=LABELS, fill_value=0
    )
    by_review.to_csv(out / "by_review_status.csv")
    lab_sub = lab.assign(
        submitters=pd.cut(
            lab["n_submitters"], [0, 1, 2, 3, 5, 1000], labels=["1", "2", "3", "4-5", "6+"]
        )
    )
    pd.crosstab(lab_sub["submitters"], lab_sub["label"]).reindex(
        columns=LABELS, fill_value=0
    ).to_csv(out / "by_submitter_count.csv")
    by_gene = pd.crosstab(lab["gene_symbol"], lab["label"]).reindex(columns=LABELS, fill_value=0)
    by_gene["total"] = by_gene.sum(axis=1)
    by_gene.sort_values("total", ascending=False).head(50).to_csv(out / "by_gene_top50.csv")
    by_gene.sort_values("downgraded", ascending=False).head(30).to_csv(
        out / "by_gene_top_downgraded.csv"
    )
    pd.crosstab(lab["consequence"], lab["label"]).reindex(columns=LABELS, fill_value=0).to_csv(
        out / "by_consequence.csv"
    )
    dg = lab[lab["label"] == "downgraded"]["latest_clnsig_class"].value_counts()
    summary = {
        "T": "2020-01",
        "latest_release": "2026-11 monthly archive (2026-11-05)",
        "label_counts": counts.to_dict(),
        "downgraded_to": dg.to_dict(),
        "downgrade_rate_among_PLP": float(
            counts["downgraded"] / max(1, counts["downgraded"] + counts["stable_pathogenic"])
        ),
        "upgrade_rate_among_VUS": float(
            counts["upgraded"] / max(1, counts["upgraded"] + counts["stable_vus"])
        ),
        "n_genes": int(lab["gene_id"].nunique()),
        "n_mondo_diseases": int(lab["mondo_id"].nunique()),
        "filter_steps": rep.steps,
    }
    write_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
