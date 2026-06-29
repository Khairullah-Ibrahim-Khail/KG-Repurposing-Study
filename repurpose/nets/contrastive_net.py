"""Q2 alternative B — end-to-end contrastive model.

Trains the therapeutic task jointly with an InfoNCE (NT-Xent) objective over
disease pairs that share at least one approved drug. No KG pretraining and no
two-phase split — this asks whether the zero-shot transfer signal can be learned
in one shot alongside the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class ContrastiveNet(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        neighbors: int = 5,
        contrastive_weight: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        node_types, _ = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.neighbors = neighbors
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

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
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def embed(self, edge_index_dict):
        h = {nt: self.drop(F.relu(self.embeddings[nt].weight)) for nt in self.node_types}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
            h = {nt: self.drop(F.relu(v)) for nt, v in h.items()}
        return {nt: self.norm(v) for nt, v in h.items()}

    def infonce_loss(self, disease_h, pos_pairs):
        """NT-Xent over a batch of positive disease pairs (in-batch negatives)."""
        if pos_pairs.size(0) == 0:
            return torch.tensor(0.0, device=disease_h.device)

        z = F.normalize(self.head(disease_h), dim=-1)
        members = pos_pairs.flatten().unique()
        z_sub = z[members]

        remap = {int(v): i for i, v in enumerate(members)}
        left = torch.tensor([remap[int(p[0])] for p in pos_pairs], device=disease_h.device)
        right = torch.tensor([remap[int(p[1])] for p in pos_pairs], device=disease_h.device)

        logits = (z_sub @ z_sub.t()) / self.temperature
        logits.fill_diagonal_(float("-inf"))
        return (F.cross_entropy(logits[left], right) + F.cross_entropy(logits[right], left)) / 2

    def zero_shot_scores(self, drug_h, query_dis, support_dis):
        q = F.normalize(self.head(query_dis), dim=-1)
        s = F.normalize(self.head(support_dis), dim=-1)
        sims = q @ s.t()
        top_sim, top_idx = sims.topk(min(self.neighbors, sims.size(1)), dim=-1)
        weights = F.softmax(top_sim, dim=-1)
        support_drug = support_dis @ drug_h.t()
        return (weights.unsqueeze(-1) * support_drug[top_idx]).sum(dim=1)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_type="drug", disease_type="disease"):
        h = self.embed(edge_index_dict)
        return (h[drug_type][drug_idx] * h[disease_type][disease_idx]).sum(-1)

    def config_dict(self):
        return {
            "architecture": "joint_contrastive",
            "hidden_dim": self.hidden_dim,
            "contrastive_weight": self.contrastive_weight,
            "temperature": self.temperature,
            "n_neighbors": self.neighbors,
            "reproduction_type": "original_ablation",
        }
