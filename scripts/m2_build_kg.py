"""M2: build the time-T heterogeneous KG and save it as PyG HeteroData.

Usage: python scripts/m2_build_kg.py
Outputs:
  data/processed/kg_T.pt                 HeteroData (torch.save)
  data/processed/kg_T_nodes/<type>.parquet   node index -> external id, name
  data/processed/kg_T_variants.parquet   variant node table (time-T attributes only)
  data/processed/kg_T_leakage_manifest.json
  results/m2/summary.json
"""

from __future__ import annotations

import pandas as pd
import torch

from kggc.data.obo import build_xref_index, parse_obo
from kggc.data.sources import SOURCES, T_DATE
from kggc.graph.build import (
    KG,
    add_clinvar,
    add_gwas,
    add_hetionet,
    add_hpo,
    add_mondo,
    read_gwas,
    read_hpo_g2p,
    to_heterodata,
)
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, seed_everything, write_json


def main() -> None:
    seed_everything(0)
    raw = RAW_DIR
    mondo = parse_obo(raw / "mondo" / "mondo_2019-11-25.obo")
    xidx = build_xref_index(mondo)
    hpo = parse_obo(raw / "hpo" / "hp_2019-11-08.obo")

    kg = KG()
    add_mondo(kg, mondo)
    print("mondo", kg.num_nodes("disease"))
    add_hetionet(
        kg,
        raw / "hetionet" / "hetionet-v1.0-nodes.tsv",
        raw / "hetionet" / "hetionet-v1.0-edges.sif.gz",
        xidx,
    )
    print("hetionet done", {k: len(v) for k, v in kg.nodes.items()})
    add_hpo(kg, hpo, read_hpo_g2p(raw / "hpo" / "genes_to_phenotype_2019-09-02.txt"))
    t_df = pd.read_parquet(PROC_DIR / "clinvar_T.parquet")
    vtab = add_clinvar(kg, t_df, xidx)
    print("clinvar variants", len(vtab))
    g = read_gwas(
        raw / "gwas" / "gwas-catalog-associations_ontology-annotated_2019-12-16.tsv", T_DATE
    )
    vtab = add_gwas(kg, g, vtab, xidx)
    print("gwas done", len(vtab))

    data = to_heterodata(kg, vtab)
    torch.save(data, PROC_DIR / "kg_T.pt")
    nd = PROC_DIR / "kg_T_nodes"
    nd.mkdir(exist_ok=True)
    for nt in kg.nodes:
        pd.DataFrame({"id": kg.nodes[nt], "name": kg.node_names[nt]}).to_parquet(
            nd / f"{nt}.parquet"
        )
    vtab.to_parquet(PROC_DIR / "kg_T_variants.parquet", index=False)

    le = pd.to_datetime(vtab["last_evaluated"], errors="coerce", format="mixed")
    manifest = {
        "T_DATE": T_DATE,
        "node_sources": data.node_source,
        "edge_sources": data.edge_source,
        "sources": {k: SOURCES[k].release_date for k in SOURCES},
        "max_clinvar_last_evaluated": str(le.max().date()) if le.notna().any() else None,
        "max_gwas_date_added": str(g["date_added"].max().date()),
        "max_gwas_pub_date": str(g["pub_date"].max().date()),
        "gwas_pmids": int(g["PUBMEDID"].nunique()),
        "variant_feature_columns": "consequence one-hot + variant type one-hot (intrinsic)",
    }
    write_json(manifest, PROC_DIR / "kg_T_leakage_manifest.json")
    summary = {
        "num_nodes": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "num_edges": {"__".join(et): int(data[et].edge_index.shape[1]) for et in data.edge_types},
        "report": kg.report,
        "gwas_rows_raw": g.attrs.get("n_raw"),
    }
    write_json(summary, RESULTS_DIR / "m2" / "summary.json")
    write_json(manifest, RESULTS_DIR / "m2" / "leakage_manifest.json")
    print(summary)


if __name__ == "__main__":
    main()
