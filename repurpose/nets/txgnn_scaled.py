"""Scaled reproduction of TxGNN (Huang et al., 2024, Nature Medicine).

DOI: 10.1038/s41591-024-03233-x

Shrunk to fit an 8 GB card; every output from this model is tagged
``scaled_reproduction``. Deviations from the published configuration:
    hidden_dim 512 -> 64   (128 already OOMs during backward)
    layers     3   -> 2
    heads      8   -> 4
    node features: learned embeddings (the paper's pre-trained features are
                   not publicly released)
    anatomy_protein_present / drug_drug edges excluded (VRAM)

The two ablation switches (``attention`` and ``affinity``) drive Q6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv


class DiseaseAffinityHead(nn.Module):
    """Metric-learning head that powers zero-shot transfer between diseases.

    An unseen disease borrows drug evidence from the most similar *seen*
    diseases, measured in a learned cosine space. Without this head the model
    has no mechanism to score drugs for a disease it never trained on.
    """

    def __init__(self, hidden_dim: int, neighbors: int = 5):
        super().__init__()
        self.neighbors = neighbors
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def cosine(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.project(query), dim=-1)
        s = F.normalize(self.project(support), dim=-1)
        return q @ s.t()

    def transfer(self, query_dis, support_dis, drug_h) -> torch.Tensor:
        """Drug scores for query diseases via their k nearest support diseases."""
        sims = self.cosine(query_dis, support_dis)
        top_sim, top_idx = sims.topk(min(self.neighbors, sims.size(1)), dim=-1)
        weights = F.softmax(top_sim, dim=-1)                 # [B, K]
        support_drug = support_dis @ drug_h.t()              # [N_support, N_drugs]
        gathered = support_drug[top_idx]                     # [B, K, N_drugs]
        return (weights.unsqueeze(-1) * gathered).sum(dim=1)

    def triplet_loss(self, disease_h, pos_pairs, neg_pairs, margin: float = 0.5):
        """Pull drug-sharing diseases together, push unrelated ones apart."""
        if pos_pairs.size(0) == 0 or neg_pairs.size(0) == 0:
            return torch.tensor(0.0, device=disease_h.device)
        z = F.normalize(self.project(disease_h), dim=-1)
        pos = (z[pos_pairs[:, 0]] * z[pos_pairs[:, 1]]).sum(-1)
        neg = (z[neg_pairs[:, 0]] * z[neg_pairs[:, 1]]).sum(-1)
        return F.relu(margin - pos).mean() + F.relu(neg + margin).mean()


class ScaledTxGNN(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        node_counts: dict,
        hidden_dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        attention: bool = True,    # Q6 switch: HGT attention vs. mean aggregation
        affinity: bool = True,     # Q6 switch: disease-affinity head on/off
        neighbors: int = 5,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.attention = attention

        self.embeddings = nn.ModuleDict(
            {nt: nn.Embedding(node_counts[nt], hidden_dim) for nt in node_types}
        )
        for table in self.embeddings.values():
            nn.init.xavier_uniform_(table.weight)

        if attention:
            self.layers = nn.ModuleList(
                [HGTConv(hidden_dim, hidden_dim, metadata, heads) for _ in range(depth)]
            )
        else:
            # Parameter-light mean aggregation (Q6 condition B).
            self.layers = nn.ModuleList(
                [
                    HeteroConv(
                        {et: SAGEConv(hidden_dim, hidden_dim, aggr="mean") for et in edge_types},
                        aggr="mean",
                    )
                    for _ in range(depth)
                ]
            )

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.affinity = DiseaseAffinityHead(hidden_dim, neighbors) if affinity else None

    def embed(self, edge_index_dict: dict) -> dict:
        h = {nt: self.drop(F.relu(self.embeddings[nt].weight)) for nt in self.node_types}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
            h = {nt: self.drop(F.relu(v)) for nt, v in h.items()}
        return {nt: self.norm(v) for nt, v in h.items()}

    @staticmethod
    def pair_score(drug_h, disease_h):
        return (drug_h * disease_h).sum(dim=-1)

    def zero_shot_scores(self, drug_h, query_dis, support_dis):
        if self.affinity is None:
            return query_dis @ drug_h.t()
        return self.affinity.transfer(query_dis, support_dis, drug_h)

    def forward(self, edge_index_dict, drug_idx, disease_idx,
                drug_type="drug", disease_type="disease"):
        h = self.embed(edge_index_dict)
        return self.pair_score(h[drug_type][drug_idx], h[disease_type][disease_idx])

    def config_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "use_attention": self.attention,
            "use_similarity": self.affinity is not None,
            "num_layers": len(self.layers),
            "scaled_from_paper": True,
            "paper_hidden_dim": 512,
            "paper_num_layers": 3,
            "paper_num_heads": 8,
            "deviation_reason": "RTX 4060 8GB VRAM constraint",
        }
