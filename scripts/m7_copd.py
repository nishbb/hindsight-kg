"""M7: multifactorial case study, COPD and emphysema.

Seeds are retrieved, not hardcoded: GWAS Catalog v2 trait search for 'chronic obstructive
pulmonary disease' and 'emphysema', then all associations for the matched trait ids.

COPD-related KG edges (time-T ClinVar variant-disease assertions) are chosen by a data rule:
  * the disease is in the MONDO 2019 subtree of a matched GWAS trait id, or
  * the disease is a respiratory system disease (MONDO:0005087 subtree) and its gene is mapped
    by a genome-wide significant COPD/emphysema association published before T.

Usage: python scripts/m7_copd.py [--agent] [--n-dossiers 12]
Outputs: results/m7/*.csv|json, results/m7/copd_subgraph.png, results/m7/dossiers/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kggc.data.cache import default_client  # noqa: E402
from kggc.data.obo import parse_obo  # noqa: E402
from kggc.utils import PROC_DIR, RAW_DIR, RESULTS_DIR, write_json  # noqa: E402

API = "https://www.ebi.ac.uk/gwas/rest/api/v2"
QUERIES = ["chronic obstructive pulmonary disease", "emphysema"]
RESP_ROOT = "MONDO:0005087"  # respiratory system disease
EXPOSURE_WORDS = ("smok", "cigarette", "tobacco", "nicotine", "air pollution", "particulate")


def resolve_traits(c) -> list[dict]:
    out = []
    for q in QUERIES:
        body = c.get_json(
            "gwas",
            f"{API}/efo-traits",
            params={"trait": q, "size": 50},
            source_version="GWAS Catalog REST v2",
        )
        for t in (body.get("_embedded") or {}).get("efo_traits", []):
            name = t["efo_trait"].lower()
            # keep disease terms; drop age-of-onset and imaging-measurement terms
            if q in name and "measurement" not in name and "age of onset" not in name:
                out.append({"query": q, "efo_id": t["efo_id"], "efo_trait": t["efo_trait"]})
    return out


def fetch_assoc(c, efo_id: str) -> list[dict]:
    rows, page = [], 0
    while True:
        body = c.get_json(
            "gwas",
            f"{API}/associations",
            params={"efo_id": efo_id, "size": 200, "page": page},
            source_version="GWAS Catalog REST v2",
        )
        items = (body.get("_embedded") or {}).get("associations", [])
        rows += items
        tp = int((body.get("page") or {}).get("totalPages", 0))
        page += 1
        if not items or page >= tp:
            return rows


def descendants(mondo, root: str) -> set[str]:
    kids: dict[str, set[str]] = {}
    for t in mondo.values():
        for p in t.is_a:
            kids.setdefault(p, set()).add(t.id)
    out, st = {root}, [root]
    while st:
        for k in kids.get(st.pop(), ()):
            if k not in out:
                out.add(k)
                st.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent", action="store_true", help="run the audit agent (config c) on the selected edges"
    )
    ap.add_argument("--n-dossiers", type=int, default=12)
    args = ap.parse_args()
    out = RESULTS_DIR / "m7"
    out.mkdir(parents=True, exist_ok=True)
    c = default_client()

    traits = resolve_traits(c)
    assoc = []
    for t in traits:
        for a in fetch_assoc(c, t["efo_id"]):
            m, e = a.get("pvalue_mantissa"), a.get("pvalue_exponent")
            log10p = (
                (np.log10(m) + e)
                if (m and e is not None)
                else np.log10(max(a.get("p_value") or 1, 1e-300))
            )
            for s in a.get("snp_allele") or []:
                assoc.append(
                    {
                        "efo_id": t["efo_id"],
                        "trait": t["efo_trait"],
                        "rsid": s.get("rs_id"),
                        "log10p": float(log10p),
                        "p_display": f"{m}e{e}" if m else str(a.get("p_value")),
                        "mapped_genes": ";".join(a.get("mapped_genes") or []),
                        "accession": a.get("accession_id"),
                        "pmid": a.get("pubmed_id"),
                    }
                )
    A = pd.DataFrame(assoc)
    A = A[A["rsid"].fillna("").str.match(r"^rs\d+$")]
    A.to_csv(out / "gwas_associations.csv", index=False)
    sig = A[A["log10p"] <= np.log10(5e-8)]
    lead = sig.sort_values("log10p").drop_duplicates("rsid")
    # one lead SNP per mapped-gene locus
    lead = lead.assign(
        locus=lead["mapped_genes"].replace("", np.nan).fillna(lead["rsid"])
    ).drop_duplicates("locus")
    lead.to_csv(out / "lead_snps.csv", index=False)

    # publication dates (for the time-T gene rule) via the dossier PubMed helper
    from kggc.variant_dossier.sources import Sources

    pm = Sources(c).pubmed_summaries(sig["pmid"].dropna().astype(str).unique().tolist())
    sig = sig.assign(
        pubdate=sig["pmid"].astype(str).map(lambda p: pm.get(p, {}).get("sortpubdate", ""))
    )
    genes_preT = sorted(
        {
            g
            for gs, d in zip(sig["mapped_genes"], sig["pubdate"])
            if d and d <= "2019-12-31"
            for g in gs.split(";")
            if g
        }
    )

    mondo = parse_obo(RAW_DIR / "mondo" / "mondo_2019-11-25.obo")
    subtree = set()
    for t in traits:
        mid = t["efo_id"].replace("_", ":")
        if mid in mondo:
            subtree |= descendants(mondo, mid)
    resp = descendants(mondo, RESP_ROOT)
    lab = pd.read_parquet(PROC_DIR / "labels.parquet")
    lab = lab[lab["label"].notna()]
    sel = lab[
        lab["mondo_id"].isin(subtree)
        | (lab["mondo_id"].isin(resp) & lab["gene_symbol"].isin(genes_preT))
    ].copy()
    sel["rule"] = np.where(
        sel["mondo_id"].isin(subtree),
        "MONDO COPD/emphysema subtree",
        "respiratory disease + COPD GWAS gene (pre-T)",
    )
    sc = pd.read_parquet(PROC_DIR / "edge_scores.parquet")[
        ["variation_id", "gnn_rgcn", "gnn_rgcn_set"]
    ]
    sel = sel.merge(sc, on="variation_id", how="left")
    keep_cols = [
        "variation_id",
        "name",
        "gene_symbol",
        "condition_name",
        "mondo_id",
        "clnsig",
        "review_status",
        "label",
        "latest_clnsig",
        "rule",
        "gnn_rgcn",
        "gnn_rgcn_set",
    ]
    sel[keep_cols].to_csv(out / "copd_edges.csv", index=False)
    print(
        f"traits {[t['efo_id'] for t in traits]}; associations {len(A)}; genome-wide significant {len(sig)}; lead loci {len(lead)}; pre-T GWAS genes {len(genes_preT)}; COPD-related edges {len(sel)}"
    )

    # dossiers for the strongest lead SNPs
    from kggc.variant_dossier.cli import run as run_dossiers

    top = lead.head(args.n_dossiers)["rsid"].tolist()
    res = run_dossiers(top, "latest", out / "dossiers")

    agent_rows = []
    if args.agent and len(sel):
        from kggc.agent.agent import AgentConfig, CachedLLM, audit_edge
        from kggc.agent.tools import TimeTTools

        tools = TimeTTools("rgcn")
        names = ["dossier", "literature_search", "kg_neighborhood", "gnn_score", "gnn_explain"]
        llm = CachedLLM(AgentConfig(tools=names))
        truth = {
            "downgraded": "downgrade",
            "upgraded": "upgrade",
            "stable_pathogenic": "keep",
            "stable_vus": "keep",
        }
        for vid, label in zip(sel["variation_id"], sel["label"]):
            r = audit_edge(llm, tools, int(vid), 6, names)
            agent_rows.append(
                {
                    "variation_id": vid,
                    "truth": truth[label],
                    "verdict": r["verdict"],
                    "confidence": r["confidence"],
                    "rationale": r["rationale"],
                    "tool_calls": r["tool_calls"],
                }
            )
        ag = pd.DataFrame(agent_rows)
        ag.to_csv(out / "agent_verdicts.csv", index=False)

    # ---------------- subgraph figure from the time-T KG
    data_nodes = {
        nt: pd.read_parquet(PROC_DIR / "kg_T_nodes" / f"{nt}.parquet")
        for nt in ("variant", "gene", "disease", "phenotype", "trait")
    }
    import torch

    data = torch.load(PROC_DIR / "kg_T.pt", weights_only=False)
    vtab = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet")
    G = nx.Graph()
    lead_in_kg = vtab[
        vtab["rsid"].isin(lead.head(60)["rsid"])
    ]  # strongest loci only, for legibility
    gene_syms = set(sel["gene_symbol"]) | set(
        lead_in_kg["gene_symbol"].replace("", np.nan).dropna()
    )
    for r in sel.itertuples():
        v = f"VCV{r.variation_id}"
        changed = r.label in ("downgraded", "upgraded")
        G.add_node(
            v,
            kind="variant",
            label=r.name.split(":")[-1][:28] if isinstance(r.name, str) else v,
            changed=changed,
        )
        G.add_node(r.gene_symbol, kind="gene", label=r.gene_symbol)
        G.add_node(r.mondo_id, kind="disease", label=r.condition_name[:30])
        G.add_edge(v, r.gene_symbol, kind="in_gene")
        G.add_edge(
            v, r.mondo_id, kind=("assertion_changed" if changed else "assertion"), label=r.label
        )
    t_names = data_nodes["trait"]
    ei = data["variant", "gwas_assoc", "trait"].edge_index.numpy()
    node_of = dict(zip(lead_in_kg["node"], lead_in_kg["rsid"]))
    for vn, tn in zip(ei[0], ei[1]):
        if vn in node_of:
            rs = node_of[vn]
            tname = str(t_names.iloc[tn]["name"])
            kind = "exposure" if any(w in tname.lower() for w in EXPOSURE_WORDS) else "trait"
            if kind == "trait" and not any(
                w in tname.lower()
                for w in (
                    "pulmonary",
                    "lung",
                    "copd",
                    "emphysema",
                    "fev",
                    "fvc",
                    "airway",
                    "spirometr",
                )
            ):
                continue
            G.add_node(rs, kind="variant", label=rs, changed=False)
            G.add_node(f"T:{tname}", kind=kind, label=tname[:30])
            G.add_edge(rs, f"T:{tname}", kind="gwas")
            g = vtab.loc[vtab["node"] == vn, "gene_symbol"].item()
            if g:
                G.add_node(g, kind="gene", label=g)
                G.add_edge(rs, g, kind="in_gene")
    # gene-phenotype edges (HPO 2019) restricted to respiratory phenotypes
    hp = parse_obo(RAW_DIR / "hpo" / "hp_2019-11-08.obo")
    resp_hp = descendants(hp, "HP:0002086")  # abnormality of the respiratory system
    gp = data["gene", "has_phenotype", "phenotype"].edge_index.numpy()
    gid2sym = dict(zip(range(len(data_nodes["gene"])), data_nodes["gene"]["name"]))
    ph = data_nodes["phenotype"]
    cnt = Counter()
    for gi, pi in zip(gp[0], gp[1]):
        sym = gid2sym[gi]
        pid = ph.iloc[pi]["id"]
        if sym in gene_syms and pid in resp_hp and cnt[sym] < 3:
            cnt[sym] += 1
            G.add_node(sym, kind="gene", label=sym)
            G.add_node(pid, kind="phenotype", label=str(ph.iloc[pi]["name"])[:28])
            G.add_edge(sym, pid, kind="has_phenotype")
    nx.write_graphml(G, out / "copd_subgraph.graphml")
    colors = {
        "variant": "#4C78A8",
        "gene": "#59A14F",
        "disease": "#E15759",
        "phenotype": "#B07AA1",
        "trait": "#F28E2B",
        "exposure": "#9C755F",
    }
    fig, ax = plt.subplots(figsize=(15, 11))
    pos = nx.kamada_kawai_layout(G)
    for kind, col in colors.items():
        ns = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=ns,
            node_color=col,
            node_size=[260 if G.nodes[n].get("changed") else 140 for n in ns],
            edgecolors=["black" if G.nodes[n].get("changed") else col for n in ns],
            linewidths=1.8,
            ax=ax,
            label=kind,
        )
    plain = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] != "assertion_changed"]
    changed = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "assertion_changed"]
    nx.draw_networkx_edges(G, pos, edgelist=plain, edge_color="#bbbbbb", width=0.8, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=changed,
        edge_color="#d62728",
        width=2.6,
        style="dashed",
        ax=ax,
        label="assertion revised by 2026",
    )
    deg = dict(G.degree())
    lab_nodes = {
        n: d["label"]
        for n, d in G.nodes(data=True)
        if d["kind"] in ("disease", "exposure", "trait")
        or (d["kind"] == "gene" and deg[n] >= 2)
        or d.get("changed")
    }
    nx.draw_networkx_labels(G, pos, labels=lab_nodes, font_size=7, ax=ax)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.set_title(
        "COPD / emphysema subgraph of the 2020-01 knowledge graph (red dashed: ClinVar assertion revised by 2026-11)",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "copd_subgraph.png", dpi=160)

    summary = {
        "traits": traits,
        "n_associations": int(len(A)),
        "n_genome_wide_significant": int(len(sig)),
        "n_lead_loci": int(len(lead)),
        "top_lead_snps": lead.head(15)[
            ["rsid", "mapped_genes", "p_display", "trait", "accession"]
        ].to_dict("records"),
        "n_preT_gwas_genes": len(genes_preT),
        "n_copd_edges": int(len(sel)),
        "copd_edges_by_rule": sel["rule"].value_counts().to_dict(),
        "copd_edges_by_label": sel["label"].value_counts().to_dict(),
        "gnn_set_by_label": sel.groupby("label")["gnn_rgcn_set"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict("index"),
        "dossiers": [s for s in res["summary"]],
        "subgraph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "kinds": Counter(d["kind"] for _, d in G.nodes(data=True)),
        },
    }
    if agent_rows:
        ag = pd.DataFrame(agent_rows)
        summary["agent"] = {
            "n": len(ag),
            "accuracy": float((ag["truth"] == ag["verdict"]).mean()),
            "verdicts": ag["verdict"].value_counts().to_dict(),
        }
    write_json(summary, out / "summary.json")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("dossiers", "top_lead_snps")},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
