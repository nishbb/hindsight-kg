"""Time-T heterogeneous knowledge graph (M2).

Every node type and edge type records the source key from ``kggc.data.sources.SOURCES``.
The leakage test asserts that every recorded source is time-safe for T.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kggc.data.clinvar import consequence_from_name, informative_conditions, split_conditions
from kggc.data.labels import map_condition
from kggc.data.obo import Term, normalize_curie

HETIONET_NODE_TYPES = {
    "Gene": "gene",
    "Biological Process": "go_bp",
    "Cellular Component": "go_cc",
    "Molecular Function": "go_mf",
    "Pathway": "pathway",
    "Anatomy": "anatomy",
    "Compound": "compound",
    "Symptom": "symptom",
    "Disease": "disease",  # mapped to MONDO
}

HETIONET_EDGES = {
    "GiG": ("gene", "interacts", "gene"),
    "GcG": ("gene", "covaries", "gene"),
    "Gr>G": ("gene", "regulates", "gene"),
    "GpBP": ("gene", "participates_bp", "go_bp"),
    "GpCC": ("gene", "participates_cc", "go_cc"),
    "GpMF": ("gene", "participates_mf", "go_mf"),
    "GpPW": ("gene", "in_pathway", "pathway"),
    "AeG": ("anatomy", "expresses", "gene"),
    "AuG": ("anatomy", "upregulates", "gene"),
    "AdG": ("anatomy", "downregulates", "gene"),
    "CbG": ("compound", "binds", "gene"),
    "CuG": ("compound", "upregulates", "gene"),
    "CdG": ("compound", "downregulates", "gene"),
    "CtD": ("compound", "treats", "disease"),
    "CpD": ("compound", "palliates", "disease"),
    "CrC": ("compound", "resembles", "compound"),
    "DaG": ("disease", "associates", "gene"),
    "DuG": ("disease", "upregulates", "gene"),
    "DdG": ("disease", "downregulates", "gene"),
    "DlA": ("disease", "localizes", "anatomy"),
    "DpS": ("disease", "presents", "symptom"),
    "DrD": ("disease", "resembles", "disease"),
}
# Skipped: CcSE (compound-side effect, 139k edges, not relevant to genotype-phenotype) and PCiC.

ASSERTION_REL = {
    "PLP": "assoc_plp",
    "VUS": "assoc_vus",
    "BLB": "assoc_blb",
    "CONFLICT": "assoc_conflict",
}
TARGET_EDGE = ("variant", "assoc_plp", "disease")

CONSEQUENCES = [
    "missense",
    "nonsense",
    "frameshift",
    "splice_site",
    "inframe_indel",
    "synonymous",
    "start_lost",
    "intronic",
    "utr5",
    "utr3",
    "structural_or_other",
    "other",
    "gwas_only",
]
VTYPES = [
    "single nucleotide variant",
    "Deletion",
    "Duplication",
    "Insertion",
    "Indel",
    "copy number loss",
    "copy number gain",
    "Microsatellite",
    "other",
]


@dataclass
class KG:
    nodes: dict[str, list[str]] = field(default_factory=dict)  # type -> list of external ids
    node_names: dict[str, list[str]] = field(default_factory=dict)
    node_source: dict[str, str] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], list[tuple[int, int]]] = field(default_factory=dict)
    edge_source: dict[tuple[str, str, str], str] = field(default_factory=dict)
    _index: dict[str, dict[str, int]] = field(default_factory=dict)
    variant_table: pd.DataFrame | None = None
    gwas_table: pd.DataFrame | None = None
    report: dict = field(default_factory=dict)

    def add_node(self, ntype: str, ext_id: str, name: str = "", source: str = "") -> int:
        idx = self._index.setdefault(ntype, {})
        if ext_id in idx:
            return idx[ext_id]
        i = len(idx)
        idx[ext_id] = i
        self.nodes.setdefault(ntype, []).append(ext_id)
        self.node_names.setdefault(ntype, []).append(name)
        if source:
            self.node_source.setdefault(ntype, source)
        return i

    def nid(self, ntype: str, ext_id: str) -> int | None:
        return self._index.get(ntype, {}).get(ext_id)

    def add_edge(self, et: tuple[str, str, str], s: int, d: int, source: str) -> None:
        self.edges.setdefault(et, []).append((s, d))
        prev = self.edge_source.get(et)
        if prev and source not in prev.split("+"):
            self.edge_source[et] = prev + "+" + source
        elif not prev:
            self.edge_source[et] = source

    def num_nodes(self, ntype: str) -> int:
        return len(self.nodes.get(ntype, []))


# --------------------------------------------------------------------------------------
def add_mondo(kg: KG, mondo: dict[str, Term]) -> None:
    for t in mondo.values():
        if t.id.startswith("MONDO:") and not t.obsolete:
            kg.add_node("disease", t.id, t.name, source="mondo_T")
    et = ("disease", "is_a", "disease")
    for t in mondo.values():
        if not t.id.startswith("MONDO:") or t.obsolete:
            continue
        c = kg.nid("disease", t.id)
        for p in t.is_a:
            pi = kg.nid("disease", p)
            if pi is not None:
                kg.add_edge(et, c, pi, "mondo_T")


def add_hpo(kg: KG, hpo: dict[str, Term], g2p: pd.DataFrame) -> None:
    for t in hpo.values():
        if t.id.startswith("HP:") and not t.obsolete:
            kg.add_node("phenotype", t.id, t.name, source="hpo_obo_T")
    et = ("phenotype", "is_a", "phenotype")
    for t in hpo.values():
        c = kg.nid("phenotype", t.id)
        if c is None:
            continue
        for p in t.is_a:
            pi = kg.nid("phenotype", p)
            if pi is not None:
                kg.add_edge(et, c, pi, "hpo_obo_T")
    et = ("gene", "has_phenotype", "phenotype")
    n_skip = 0
    for gid, sym, hp in zip(g2p["gene_id"], g2p["gene_symbol"], g2p["hpo_id"]):
        p = kg.nid("phenotype", hp)
        if p is None:
            n_skip += 1
            continue
        g = kg.add_node("gene", f"Gene::{gid}", sym, source="hetionet")
        kg.add_edge(et, g, p, "hpo_g2p_T")
    kg.report["hpo_g2p_unmapped_terms"] = n_skip


def read_hpo_g2p(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["gene_id", "gene_symbol", "hpo_name", "hpo_id"],
    )
    return df.drop_duplicates(["gene_id", "hpo_id"])


def add_hetionet(
    kg: KG, nodes_path: str | Path, edges_path: str | Path, xref_index: dict[str, str]
) -> None:
    nodes = pd.read_csv(nodes_path, sep="\t", dtype=str)
    doid_to_mondo: dict[str, str] = {}
    unmapped = 0
    for nid, name, kind in zip(nodes["id"], nodes["name"], nodes["kind"]):
        nt = HETIONET_NODE_TYPES.get(kind)
        if nt is None:
            continue
        if nt == "disease":
            doid = nid.split("::", 1)[1]
            m = xref_index.get(normalize_curie(doid))
            if m is None or kg.nid("disease", m) is None:
                unmapped += 1
                continue
            doid_to_mondo[nid] = m
            continue
        kg.add_node(nt, nid, name, source="hetionet")
    kg.report["hetionet_diseases_mapped_to_mondo"] = len(doid_to_mondo)
    kg.report["hetionet_diseases_unmapped"] = unmapped

    opener = gzip.open if str(edges_path).endswith(".gz") else open
    with opener(edges_path, "rt") as fh:
        next(fh)
        for line in fh:
            s, me, d = line.rstrip("\n").split("\t")
            et = HETIONET_EDGES.get(me)
            if et is None:
                continue
            si = _het_lookup(kg, et[0], s, doid_to_mondo)
            di = _het_lookup(kg, et[2], d, doid_to_mondo)
            if si is None or di is None:
                continue
            kg.add_edge(et, si, di, "hetionet")


def _het_lookup(kg: KG, ntype: str, hid: str, doid_to_mondo: dict[str, str]) -> int | None:
    if ntype == "disease":
        m = doid_to_mondo.get(hid)
        return kg.nid("disease", m) if m else None
    return kg.nid(ntype, hid)


def _vtype(t: str) -> str:
    return t if t in VTYPES else "other"


def add_clinvar(kg: KG, t_df: pd.DataFrame, xref_index: dict[str, str]) -> pd.DataFrame:
    """Variant nodes and time-T assertion edges. Returns the variant node table."""
    d = t_df[
        (t_df["origin_simple"] == "germline")
        & (t_df["clnsig_class"].isin(list(ASSERTION_REL)))
        & (t_df["gene_id"] > 0)
    ]
    rows = []
    for r in d.itertuples(index=False):
        conds = informative_conditions(split_conditions(r.phenotype_ids, r.phenotype_list))
        mondos = sorted({m for _, xr in conds if (m := map_condition(xr, xref_index))})
        if not mondos:
            continue
        rows.append((r, mondos))
    recs = []
    et_gene = ("variant", "in_gene", "gene")
    for r, mondos in rows:
        v = kg.add_node("variant", f"VCV:{int(r.variation_id)}", r.name, source="clinvar_T")
        g = kg.add_node("gene", f"Gene::{r.gene_id}", r.gene_symbol, source="hetionet")
        kg.add_edge(et_gene, v, g, "clinvar_T")
        rel = ASSERTION_REL[r.clnsig_class]
        for m in mondos:
            di = kg.nid("disease", m)
            kg.add_edge(("variant", rel, "disease"), v, di, "clinvar_T")
        recs.append(
            {
                "node": v,
                "variation_id": int(r.variation_id),
                "rsid": r.rsid,
                "gene_id": int(r.gene_id),
                "gene_symbol": r.gene_symbol,
                "clnsig_class_T": r.clnsig_class,
                "n_mondo": len(mondos),
                "consequence": consequence_from_name(r.name, r.type),
                "vtype": _vtype(r.type),
                "last_evaluated": r.last_evaluated,
                "origin": "clinvar",
            }
        )
    kg.report["clinvar_variant_nodes"] = len(recs)
    return pd.DataFrame(recs)


def read_gwas(path: str | Path, t_date: str, p_thresh: float = 5e-8) -> pd.DataFrame:
    g = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, na_filter=False)
    g["date_added"] = pd.to_datetime(g["DATE ADDED TO CATALOG"], errors="coerce")
    g["pub_date"] = pd.to_datetime(g["DATE"], errors="coerce")
    g["p"] = pd.to_numeric(g["P-VALUE"], errors="coerce")
    n0 = len(g)
    g = g[(g["date_added"] <= t_date) & (g["pub_date"] <= t_date) & (g["p"] <= p_thresh)]
    g = g[g["SNP_ID_CURRENT"].str.fullmatch(r"\d+")]
    g = g.assign(rsid="rs" + g["SNP_ID_CURRENT"])
    g.attrs["n_raw"] = n0
    return g


def _trait_curie(uri: str) -> str:
    tail = uri.strip().rsplit("/", 1)[-1]
    pre, _, local = tail.partition("_")
    return normalize_curie(f"{pre}:{local}") if local else tail


def add_gwas(
    kg: KG, g: pd.DataFrame, vtab: pd.DataFrame, xref_index: dict[str, str]
) -> pd.DataFrame:
    sym_to_gene = {}
    for ext, name in zip(kg.nodes.get("gene", []), kg.node_names.get("gene", [])):
        sym_to_gene.setdefault(name, ext)
    rs_to_node = {rs: n for rs, n in zip(vtab["rsid"], vtab["node"]) if rs}
    new_recs = []
    et_va = ("variant", "gwas_assoc", "trait")
    et_tm = ("trait", "maps_to", "disease")
    seen_pairs = set()
    for rs, mapped_gene, uris, traits in zip(
        g["rsid"],
        g["MAPPED_GENE"],
        g["MAPPED_TRAIT_URI"],
        g["MAPPED_TRAIT"],
    ):
        if not uris:
            continue
        v = rs_to_node.get(rs)
        if v is None:
            v = kg.add_node("variant", f"RS:{rs}", rs, source="gwas_T")
            rs_to_node[rs] = v
            genes = [x.strip() for x in mapped_gene.split(",")] if mapped_gene else []
            genes = [x for x in genes if " - " not in x and x]
            gid = None
            if len(genes) == 1 and genes[0] in sym_to_gene:
                gid = sym_to_gene[genes[0]]
                kg.add_edge(("variant", "in_gene", "gene"), v, kg.nid("gene", gid), "gwas_T")
            new_recs.append(
                {
                    "node": v,
                    "variation_id": -1,
                    "rsid": rs,
                    "gene_id": int(gid.split("::")[1]) if gid else -1,
                    "gene_symbol": genes[0] if len(genes) == 1 else "",
                    "clnsig_class_T": "",
                    "n_mondo": 0,
                    "consequence": "gwas_only",
                    "vtype": "single nucleotide variant",
                    "last_evaluated": "",
                    "origin": "gwas",
                }
            )
        tnames = [x.strip() for x in traits.split(",")]
        for i, u in enumerate(uris.split(",")):
            curie = _trait_curie(u)
            tname = tnames[i] if i < len(tnames) else ""
            t = kg.add_node("trait", curie, tname, source="gwas_T")
            if (v, t) not in seen_pairs:
                seen_pairs.add((v, t))
                kg.add_edge(et_va, v, t, "gwas_T")
            m = curie if curie.startswith("MONDO:") else xref_index.get(curie)
            di = kg.nid("disease", m) if m else None
            if di is not None and (t, di) not in seen_pairs:
                seen_pairs.add((t, di))
                kg.add_edge(et_tm, t, di, "mondo_T")
    kg.report["gwas_rows_used"] = len(g)
    kg.report["gwas_only_variant_nodes"] = len(new_recs)
    return pd.concat([vtab, pd.DataFrame(new_recs)], ignore_index=True)


def variant_features(vtab: pd.DataFrame) -> np.ndarray:
    """Intrinsic, time-safe variant features: consequence and variant type one-hots."""
    vt = vtab.sort_values("node")
    c = np.zeros((len(vt), len(CONSEQUENCES)), dtype=np.float32)
    t = np.zeros((len(vt), len(VTYPES)), dtype=np.float32)
    for i, (cons, typ) in enumerate(zip(vt["consequence"], vt["vtype"])):
        c[i, CONSEQUENCES.index(cons if cons in CONSEQUENCES else "other")] = 1
        t[i, VTYPES.index(typ if typ in VTYPES else "other")] = 1
    return np.concatenate([c, t], axis=1)


def to_heterodata(kg: KG, vtab: pd.DataFrame):
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for nt, ids in kg.nodes.items():
        data[nt].num_nodes = len(ids)
    data["variant"].x = torch.from_numpy(variant_features(vtab))
    for et, pairs in kg.edges.items():
        arr = np.unique(np.asarray(pairs, dtype=np.int64), axis=0).T
        data[et].edge_index = torch.from_numpy(arr)
    data.edge_source = {"__".join(k): v for k, v in kg.edge_source.items()}
    data.node_source = dict(kg.node_source)
    return data
