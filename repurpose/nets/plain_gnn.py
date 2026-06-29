"""Plain heterogeneous GNN baseline — the "GNN" column of the Q5 table.

Two HGT layers over PrimeKG, no disease-affinity head and no metric learning.
Nodes have no external features, so each node type gets its own learnable
embedding table (Xavier init); feeding raw integer ids as floats blows up the
gradients, hence embeddings.

Setting ``depth=0`` turns message passing off entirely, giving the "no-KG"
control used for Q1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class PlainHGT(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        node_types, _ = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim

        self.embeddings = nn.ModuleDict(
            {nt: nn.Embedding(node_counts[nt], hidden_dim) for nt in node_types}
        )
        for table in self.embeddings.values():
            nn.init.xavier_uniform_(table.weight)

        self.layers = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, heads) for _ in range(depth)]
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def embed(self, edge_index_dict: dict) -> dict:
        """Full-graph forward pass returning per-type node embeddings."""
        h = {nt: self.drop(F.relu(self.embeddings[nt].weight)) for nt in self.node_types}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
            h = {nt: self.drop(F.relu(v)) for nt, v in h.items()}
        return {nt: self.norm(v) for nt, v in h.items()}

    @staticmethod
    def pair_score(drug_h: torch.Tensor, disease_h: torch.Tensor) -> torch.Tensor:
        return (drug_h * disease_h).sum(dim=-1)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_type="drug", disease_type="disease"):
        h = self.embed(edge_index_dict)
        return self.pair_score(h[drug_type][drug_idx], h[disease_type][disease_idx])
