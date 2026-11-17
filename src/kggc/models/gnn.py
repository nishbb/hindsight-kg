"""Heterogeneous GNN link predictors for the target relation (variant, assoc_plp, disease).

Two encoders share one interface:
* ``rgcn``: relational GCN in the sense of Schlichtkrull et al. (one weight matrix per relation,
  mean-normalised neighbour aggregation, separate self-loop weight), implemented as a PyG
  ``HeteroConv`` of ``SAGEConv`` (no basis decomposition).
* ``hgt``: Heterogeneous Graph Transformer (``HGTConv``).

Leak-free message passing design (see NOTES.md, M3):
* assertion edges only point variant -> disease, so a variant never receives a message that
  encodes its own classification;
* the edges being scored are never in the message graph;
* per-gene summaries of other variants' assertions enter through derived gene-disease edges
  that are recomputed from the current message edges only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv

ASSERT_TYPES = ("plp", "blb", "vus", "conflict")
DERIVED_TYPES = ("plp", "blb", "vus")


def derived_gene_disease(edge_index: torch.Tensor, variant_gene: torch.Tensor) -> torch.Tensor:
    """(variant -> disease) edges to unique (gene -> disease) edges; variants without gene dropped."""
    g = variant_gene[edge_index[0]]
    keep = g >= 0
    pairs = torch.stack([g[keep], edge_index[1][keep]])
    if pairs.numel() == 0:
        return pairs
    return torch.unique(pairs, dim=1)


def message_edge_dict(
    static: dict, assertion: dict[str, torch.Tensor], variant_gene: torch.Tensor
) -> dict:
    """Combine static background edges with the current assertion message edges."""
    out = dict(static)
    for k in ASSERT_TYPES:
        ei = assertion.get(k)
        if ei is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        out[("variant", f"assoc_{k}", "disease")] = ei
        if k in DERIVED_TYPES:
            d = derived_gene_disease(ei, variant_gene)
            out[("gene", f"has_{k}_variant_for", "disease")] = d
            out[("disease", f"rev_has_{k}_variant_for", "gene")] = d.flip(0)
    return out


def all_edge_types(static_types: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    types = list(static_types)
    for k in ASSERT_TYPES:
        types.append(("variant", f"assoc_{k}", "disease"))
        if k in DERIVED_TYPES:
            types.append(("gene", f"has_{k}_variant_for", "disease"))
            types.append(("disease", f"rev_has_{k}_variant_for", "gene"))
    return types


class HeteroLinkPredictor(nn.Module):
    def __init__(
        self,
        num_nodes: dict[str, int],
        variant_feat_dim: int,
        edge_types: list[tuple[str, str, str]],
        hidden: int = 64,
        layers: int = 2,
        encoder: str = "rgcn",
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.node_types = list(num_nodes)
        self.emb = nn.ModuleDict(
            {nt: nn.Embedding(n, hidden) for nt, n in num_nodes.items() if nt != "variant"}
        )
        self.variant_in = nn.Sequential(nn.Linear(variant_feat_dim, hidden), nn.ReLU())
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if encoder == "rgcn":
                conv = HeteroConv(
                    {et: SAGEConv((hidden, hidden), hidden, aggr="mean") for et in edge_types},
                    aggr="sum",
                )
            elif encoder == "hgt":
                conv = HGTConv(hidden, hidden, (self.node_types, edge_types), heads=heads)
            else:
                raise ValueError(encoder)
            self.convs.append(conv)
        self.dec = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def encode(self, variant_x: torch.Tensor, edge_index_dict: dict) -> dict[str, torch.Tensor]:
        x = {nt: e.weight for nt, e in self.emb.items()}
        x["variant"] = self.variant_in(variant_x)
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index_dict)
            # node types that receive no message keep their previous representation
            new = {}
            for nt in x:
                if nt in h and h[nt] is not None:
                    v = h[nt] + x[nt]  # residual
                    if i < len(self.convs) - 1:
                        v = F.dropout(F.relu(v), p=self.dropout, training=self.training)
                    new[nt] = v
                else:
                    new[nt] = x[nt]
            x = new
        return x

    def decode(self, h: dict[str, torch.Tensor], pairs: torch.Tensor) -> torch.Tensor:
        hv, hd = h["variant"][pairs[0]], h["disease"][pairs[1]]
        return self.dec(torch.cat([hv, hd, hv * hd], dim=-1)).squeeze(-1)
