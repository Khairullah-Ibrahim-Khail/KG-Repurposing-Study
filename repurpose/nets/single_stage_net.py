"""Q2 alternative A — single-stage multi-task model.

No Phase 1 / Phase 2 separation: the KG link-prediction loss and the
therapeutic loss are optimized together from the first epoch. Same HGT encoder
as the scaled TxGNN, and it keeps the disease-affinity head so zero-shot scoring
still works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from repurpose.nets.txgnn_scaled import DiseaseAffinityHead


class JointTaskNet(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        affinity: bool = True,
        neighbors: int = 5,
        kg_weight: float = 0.5,
    ):
        super().__init__()
        node_types, _ = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.kg_weight = kg_weight

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
        self.affinity = DiseaseAffinityHead(hidden_dim, neighbors) if affinity else None

    def embed(self, edge_index_dict):
        h = {nt: self.drop(F.relu(self.embeddings[nt].weight)) for nt in self.node_types}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
            h = {nt: self.drop(F.relu(v)) for nt, v in h.items()}
        return {nt: self.norm(v) for nt, v in h.items()}

    @staticmethod
    def pair_score(drug_h, disease_h):
        return (drug_h * disease_h).sum(-1)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_type="drug", disease_type="disease"):
        h = self.embed(edge_index_dict)
        return self.pair_score(h[drug_type][drug_idx], h[disease_type][disease_idx])

    def config_dict(self):
        return {
            "architecture": "single_stage",
            "hidden_dim": self.hidden_dim,
            "kg_loss_weight": self.kg_weight,
            "use_similarity": self.affinity is not None,
            "reproduction_type": "original_ablation",
        }
