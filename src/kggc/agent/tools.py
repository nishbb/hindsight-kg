"""Time-T evidence tools for the audit agent.

Every tool reads only time-T artifacts (the 2020-01 graph, out-of-fold GNN scores that were
fitted without outcome labels, ClinVar releases up to 2020-01, literature published before T).
Each tool result carries an ``evidence_id`` the agent must cite in its rationale.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd
import torch

from kggc.data.cache import default_client
from kggc.utils import PROC_DIR
from kggc.variant_dossier.build import DossierBuilder

AS_OF = "2020-01-01"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Fields of the time-T ClinVar record shown to the agent. Nothing derived from later releases.
TIME_T_FIELDS = [
    "variation_id",
    "name",
    "gene_symbol",
    "clnsig",
    "review_status",
    "stars",
    "n_submitters",
    "last_evaluated",
    "condition_name",
    "mondo_id",
    "consequence",
    "ever_conflicting_before_T",
    "class_changed_before_T",
]


class TimeTTools:
    def __init__(self, gnn_tag: str = "rgcn", alpha: float = 0.1) -> None:
        from kggc.models.conformal import calibrate_by_fold

        self.data = torch.load(PROC_DIR / "kg_T.pt", weights_only=False)
        self.vtab = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet").set_index("node")
        self.names = {
            nt: pd.read_parquet(PROC_DIR / "kg_T_nodes" / f"{nt}.parquet")
            for nt in self.data.node_types
        }
        self.ext2node = {nt: dict(zip(df["id"], df.index)) for nt, df in self.names.items()}
        lab = pd.read_parquet(PROC_DIR / "labels.parquet")
        ann = pd.read_parquet(PROC_DIR / "annotations_T.parquet")
        lab = lab.merge(
            ann[["variation_id", "gnomad_af", "in_gnomad"]], on="variation_id", how="left"
        )
        self.records = lab.set_index("variation_id")
        sc = pd.read_parquet(PROC_DIR / f"gnn_scores_{gnn_tag}.parquet")
        cal = pd.read_parquet(PROC_DIR / f"gnn_calibration_{gnn_tag}.parquet")
        self.scores = calibrate_by_fold(sc, cal, alpha).set_index(["variation_id", "mondo_id"])
        self.alpha = alpha
        # assertion table: variant node, disease node, class
        rows = []
        for cls in ("plp", "vus", "blb", "conflict"):
            ei = self.data["variant", f"assoc_{cls}", "disease"].edge_index
            rows.append(pd.DataFrame({"v": ei[0].numpy(), "d": ei[1].numpy(), "cls": cls}))
        a = pd.concat(rows, ignore_index=True)
        a["gene"] = self.vtab["gene_id"].reindex(a["v"]).values
        a["cons"] = self.vtab["consequence"].reindex(a["v"]).values
        self.assert_tab = a
        self.by_gene = {g: grp for g, grp in a.groupby("gene")}
        # disease hierarchy and Hetionet disease-gene associations
        isa = self.data["disease", "is_a", "disease"].edge_index.numpy()
        self.parents, self.children = defaultdict(set), defaultdict(set)
        for c, p in zip(isa[0], isa[1]):
            self.parents[int(c)].add(int(p))
            self.children[int(p)].add(int(c))
        dag = self.data["disease", "associates", "gene"].edge_index.numpy()
        self.het_dg = set(zip(dag[0].tolist(), dag[1].tolist()))
        self._adj = None
        self._dossier = None
        self._dossier_cache: dict[int, dict] = {}
        self.http = default_client()

    # ------------------------------------------------------------------ helpers
    def _dname(self, d: int) -> str:
        return f"{self.names['disease'].iloc[d]['name']} ({self.names['disease'].iloc[d]['id']})"

    def _vname(self, v: int) -> str:
        return str(self.names["variant"].iloc[v]["name"])

    def _gene_node(self, gene_id: int) -> int | None:
        return self.ext2node["gene"].get(f"Gene::{gene_id}")

    def _edge(self, variation_id: int) -> tuple[int, int, str, int]:
        r = self.records.loc[variation_id]
        v = self.ext2node["variant"][f"VCV:{variation_id}"]
        d = self.ext2node["disease"][r["mondo_id"]]
        return v, d, r["mondo_id"], int(r["gene_id"])

    # ------------------------------------------------------------------ tools
    def clinvar_record(self, variation_id: int) -> dict:
        r = self.records.loc[int(variation_id)]
        out = {
            k: (r[k].item() if hasattr(r[k], "item") else r[k])
            for k in TIME_T_FIELDS
            if k != "variation_id"
        }
        out["variation_id"] = int(variation_id)
        out["gnomad_v2.1.1_af"] = float(r["gnomad_af"]) if pd.notna(r["gnomad_af"]) else None
        out["evidence_id"] = f"ClinVar2020-01:VCV{int(variation_id)}"
        return out

    def gnn_score(self, variation_id: int) -> dict:
        vid = int(variation_id)
        mondo = self.records.loc[vid, "mondo_id"]
        s = self.scores.loc[(vid, mondo)]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        return {
            "evidence_id": f"GNN:VCV{vid}",
            "edge": f"VCV:{vid} -> {mondo}",
            "plausibility_logit": round(float(s["score"]), 3),
            "conformal_p_value_pathogenic_like": round(float(s["p_plp"]), 4),
            "conformal_p_value_benign_like": round(float(s["p_blb"]), 4),
            "conformal_set_at_alpha_0.1": s["conformal_set"],
            "meaning": (
                "Out-of-fold R-GCN score for the pathogenic relation, fitted on the 2020-01 graph "
                "without this variant's own assertion. The conformal set says which time-T class "
                "(pathogenic-like 'plp' or benign-like 'blb') the edge resembles; 'empty' means it "
                "resembles neither well, 'both' means ambiguous."
            ),
        }

    def gnn_explain(self, variation_id: int, k: int = 4) -> dict:
        """Metapath evidence around the edge: supporting and contradicting graph paths."""
        vid = int(variation_id)
        v, d, mondo, gene_id = self._edge(vid)
        g = self.by_gene.get(gene_id, self.assert_tab.iloc[0:0])
        g = g[g["v"] != v]
        cons = self.vtab.loc[v, "consequence"]
        near = self.parents[d] | self.children[d]
        gnode = self._gene_node(gene_id)
        sup, con = [], []
        same = g[(g["d"] == d) & (g["cls"] == "plp")]
        if len(same):
            sup.append(
                {
                    "path": "variant -in_gene-> gene <-in_gene- other variant -assoc_plp-> SAME disease",
                    "count": int(len(same)),
                    "examples": [self._vname(x) for x in same["v"].head(k)],
                }
            )
        rel = g[g["d"].isin(near) & (g["cls"] == "plp")]
        if len(rel):
            sup.append(
                {
                    "path": "other P/LP variants in the gene for a parent or child disease",
                    "count": int(len(rel)),
                    "diseases": sorted({self._dname(x) for x in rel["d"]})[:k],
                }
            )
        if gnode is not None and (
            (d, gnode) in self.het_dg or any((p, gnode) in self.het_dg for p in self.parents[d])
        ):
            sup.append(
                {
                    "path": "Hetionet 2017 disease-associates-gene edge for this disease or its parent",
                    "count": 1,
                }
            )
        sc = g[(g["cons"] == cons) & (g["cls"] == "plp")]
        if len(sc):
            sup.append(
                {
                    "path": f"other P/LP variants in the gene with the same consequence ({cons})",
                    "count": int(len(sc)),
                }
            )
        other = g[(g["cls"] == "plp") & ~g["d"].isin(near | {d})]
        if len(other):
            top = Counter(other["d"]).most_common(k)
            con.append(
                {
                    "path": "gene's P/LP variants point to OTHER diseases",
                    "count": int(len(other)),
                    "top_diseases": [f"{self._dname(x)}: {n}" for x, n in top],
                }
            )
        blb = g[(g["cons"] == cons) & (g["cls"] == "blb")]
        if len(blb):
            con.append(
                {
                    "path": f"benign/likely benign variants in the gene with the same consequence ({cons})",
                    "count": int(len(blb)),
                    "examples": [self._vname(x) for x in blb["v"].head(k)],
                }
            )
        cf = g[g["cls"] == "conflict"]
        if len(cf):
            con.append(
                {
                    "path": "variants in the gene with conflicting interpretations at T",
                    "count": int(len(cf)),
                }
            )
        tally = g["cls"].value_counts().to_dict()
        if not len(same) and not len(rel):
            con.append(
                {
                    "path": "no other P/LP variant in this gene for this disease or a related disease",
                    "count": 0,
                }
            )
        return {
            "evidence_id": f"GRAPH:VCV{vid}",
            "edge": f"{self._vname(v)} -> {self._dname(d)}",
            "gene_assertion_counts_at_T": {k_: int(n) for k_, n in tally.items()},
            "supporting_paths": sup,
            "contradicting_paths": con,
            "note": "Counts exclude the audited variant. Built from the 2020-01 graph only.",
        }

    def kg_neighborhood(self, node: str, hops: int = 1, k: int = 8) -> dict:
        """Neighbour summary for 'VCV:<id>', 'Gene::<entrez>' or a MONDO id (hops 1 or 2)."""
        if self._adj is None:
            self._adj = defaultdict(list)
            for s, r, t in self.data.edge_types:
                ei = self.data[s, r, t].edge_index.numpy()
                df = pd.DataFrame({"a": ei[0], "b": ei[1]})
                for a_, grp in df.groupby("a"):
                    self._adj[(s, int(a_))].append((r, t, grp["b"].to_numpy()))
                for b_, grp in df.groupby("b"):
                    self._adj[(t, int(b_))].append((f"rev_{r}", s, grp["a"].to_numpy()))
        nt = (
            "variant"
            if node.startswith(("VCV:", "RS:"))
            else ("gene" if node.startswith("Gene::") else "disease")
        )
        idx = self.ext2node[nt].get(node)
        if idx is None:
            return {"evidence_id": f"KG:{node}", "error": f"node {node} not in the 2020-01 graph"}
        out = {
            "evidence_id": f"KG:{node}",
            "node": node,
            "name": str(self.names[nt].iloc[idx]["name"]),
            "neighbors": {},
        }
        for r, t, nb in self._adj[(nt, idx)]:
            key = f"{r} -> {t}"
            names = [str(self.names[t].iloc[int(x)]["name"])[:60] for x in nb[:k]]
            out["neighbors"][key] = {"count": int(len(nb)), "examples": names}
        if hops >= 2:
            two = Counter()
            for r, t, nb in self._adj[(nt, idx)]:
                for x in nb[:200]:
                    for r2, t2, nb2 in self._adj[(t, int(x))]:
                        two[f"{r} -> {t} -> {r2} -> {t2}"] += len(nb2)
            out["two_hop_counts"] = dict(two.most_common(12))
        return out

    def dossier(self, variation_id: int) -> dict:
        """Compact variant dossier as of 2020-01-01 (ClinVar <= 2020-01, gnomAD v2.1.1, GWAS and literature published before T)."""
        vid = int(variation_id)
        if vid in self._dossier_cache:
            return self._dossier_cache[vid]
        out = self._dossier_uncached(vid)
        self._dossier_cache[vid] = out
        return out

    def _dossier_uncached(self, vid: int) -> dict:
        base = self.clinvar_record(vid)
        rs = self.records.loc[vid, "rsid"]
        out = {"evidence_id": f"DOSSIER2020:VCV{vid}", "time_T_clinvar_record": base}
        if not rs:
            out["note"] = "No dbSNP rsID at T; only the ClinVar record is available."
            return out
        if self._dossier is None:
            self._dossier = DossierBuilder()
        try:
            d = self._dossier.build(rs, AS_OF, max_abstracts=2)
        except Exception as err:
            out["error"] = f"dossier failed: {type(err).__name__}"
            return out
        mine = [c for c in d["clinvar"] if str(c["variation_id"]) == str(vid)] or d["clinvar"]
        out.update(
            {
                "rsid": d["rsid_current"],
                "alleles": [
                    {
                        "hgvs": a["hgvs_g"],
                        "type": a["type"],
                        "gnomad_v2.1.1": (a.get("frequency") or {}).get("gnomad_v2"),
                        "vep": a.get("vep"),
                    }
                    for a in d["alleles"]
                ],
                "clinvar_at_T": [
                    {
                        k: c[k]
                        for k in (
                            "variation_id",
                            "classification",
                            "review_status",
                            "n_submitters",
                            "conditions",
                            "submission_classes",
                        )
                    }
                    | {"history": [(h["snapshot"], h["clnsig"]) for h in c["history"]]}
                    for c in mine
                ],
                "gwas_published_before_T": {
                    k: d["gwas"][k]
                    for k in ("n_associations_in_scope", "n_genome_wide_significant", "top_traits")
                },
                "literature_before_T": {
                    "n_articles": d["literature"]["n_in_scope"],
                    "top_pmids": d["literature"]["top"],
                },
                "conflicts_detected": [k["type"] for k in d["conflicts"]],
            }
        )
        titles = {
            rid: r["data"].get("title")
            for rid, r in d["records"].items()
            if rid.startswith("PMID:")
        }
        out["top_article_titles"] = {k: v for k, v in list(titles.items())[:3]}
        return out

    def literature_search(self, query: str, max_results: int = 5) -> dict:
        """PubMed search restricted to articles published before T."""
        rec = self.http.request(
            "ncbi",
            f"{EUTILS}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "datetype": "pdat",
                "mindate": "1900/01/01",
                "maxdate": "2019/12/31",
                "sort": "relevance",
            },
            source_version="PubMed E-utilities (maxdate 2019/12/31)",
        )
        body = rec["body"] if isinstance(rec["body"], dict) else {}
        ids = body.get("esearchresult", {}).get("idlist", [])
        count = body.get("esearchresult", {}).get("count", "0")
        from kggc.variant_dossier.sources import Sources

        meta = Sources(self.http).pubmed_summaries(ids)
        return {
            "evidence_id": f"PUBMED_SEARCH:{query[:60]}",
            "n_hits_before_T": int(count),
            "articles": [
                {
                    "pmid": f"PMID:{p}",
                    "title": meta.get(p, {}).get("title", ""),
                    "date": meta.get(p, {}).get("sortpubdate", ""),
                }
                for p in ids
                if meta.get(p, {}).get("sortpubdate", "9999") <= "2019-12-31"
            ],
        }


TOOL_SPECS = {
    "kg_neighborhood": {
        "description": "Summarize the 2020-01 knowledge-graph neighbourhood of a node. Node ids look like 'VCV:12345', 'Gene::7157' (Entrez) or 'MONDO:0007254'.",
        "parameters": {
            "type": "object",
            "properties": {"node": {"type": "string"}, "hops": {"type": "integer", "enum": [1, 2]}},
            "required": ["node"],
        },
    },
    "gnn_score": {
        "description": "Out-of-fold graph neural network plausibility score and split-conformal set for the audited variant-disease edge.",
        "parameters": {
            "type": "object",
            "properties": {"variation_id": {"type": "integer"}},
            "required": ["variation_id"],
        },
    },
    "gnn_explain": {
        "description": "Top supporting and contradicting graph paths (metapaths) around the audited edge in the 2020-01 graph.",
        "parameters": {
            "type": "object",
            "properties": {"variation_id": {"type": "integer"}},
            "required": ["variation_id"],
        },
    },
    "dossier": {
        "description": "Evidence dossier for a variant as of 2020-01-01: ClinVar record, submissions and history up to 2020-01, gnomAD v2.1.1 frequencies, VEP consequence, GWAS and literature published before 2020.",
        "parameters": {
            "type": "object",
            "properties": {"variation_id": {"type": "integer"}},
            "required": ["variation_id"],
        },
    },
    "literature_search": {
        "description": "Search PubMed for articles published before 2020-01-01. Returns titles and PMIDs.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
GNN_TOOLS = {"gnn_score", "gnn_explain"}


def openai_tools(names: list[str]) -> list[dict]:
    return [{"type": "function", "function": {"name": n, **TOOL_SPECS[n]}} for n in names]


def run_tool(tools: TimeTTools, name: str, args: dict) -> str:
    try:
        fn = getattr(tools, name)
        res = fn(**args)
    except Exception as err:  # the agent sees the error and can recover
        res = {"error": f"{type(err).__name__}: {err}"[:300]}
    return json.dumps(res, default=str)[:6000]
