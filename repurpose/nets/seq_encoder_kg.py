"""Q1 KG-augmented sequence encoder (transformer + neighbor-triple context).

A small transformer encoder scores a (drug, disease) pair. The KG-augmented
variant prepends up to K neighbor triples — each embedded as
(head || relation || tail) and projected back to the model width — as context
tokens. Compared against ``seq_encoder_plain`` (same backbone, no context) this
isolates the effect of KG structure in a non-GNN model.
"""

import torch
import torch.nn as nn


class TripleTokenizer(nn.Module):
    """Turn a batch of neighbor triples into one context token each."""

    def __init__(self, n_relations: int, hidden_dim: int):
        super().__init__()
        self.rel_emb = nn.Embedding(n_relations, hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, head_h, rel_ids, tail_h):
        r = self.rel_emb(rel_ids)
        return self.fuse(torch.cat([head_h, r, tail_h], dim=-1))


class KGSeqEncoder(nn.Module):
    def __init__(self, n_drugs, n_diseases, n_relations,
                 hidden_dim=128, heads=4, depth=2, dropout=0.1, max_neighbors=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.use_kg = True

        self.drug_emb = nn.Embedding(n_drugs, hidden_dim)
        self.disease_emb = nn.Embedding(n_diseases, hidden_dim)
        self.tokenizer = TripleTokenizer(n_relations, hidden_dim)

        self.drug_token = nn.Parameter(torch.randn(hidden_dim))
        self.disease_token = nn.Parameter(torch.randn(hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.scorer = nn.Linear(hidden_dim * 2, 1)

    def _encode(self, base_h, type_token, nbr):
        seq = (base_h + type_token.unsqueeze(0)).unsqueeze(1)  # [B, 1, D]
        if nbr is not None and nbr.get("head") is not None and nbr["head"].size(1) > 0:
            ctx = self.tokenizer(nbr["head"], nbr["rel"], nbr["tail"])
            seq = torch.cat([seq, ctx], dim=1)
        return self.encoder(seq)[:, 0, :]

    def forward(self, drug_idx, disease_idx, drug_nbr=None, disease_nbr=None):
        drug_h = self._encode(self.drug_emb(drug_idx), self.drug_token, drug_nbr)
        disease_h = self._encode(self.disease_emb(disease_idx), self.disease_token, disease_nbr)
        return self.scorer(torch.cat([drug_h, disease_h], dim=-1)).squeeze(-1)


def neighbor_lookup(edges, entity_idx, relation_idx, max_neighbors=8,
                    drug_type="drug", disease_type="disease"):
    """Precompute per-entity neighbor triples as (head_idx, rel_id, tail_idx)."""
    drug_map = entity_idx.get(drug_type, {})
    disease_map = entity_idx.get(disease_type, {})
    drug_nbrs, disease_nbrs = {}, {}

    for _, row in edges.iterrows():
        rel = row["relation"]
        if rel not in relation_idx:
            continue
        rid = relation_idx[rel]

        if row["x_type"] == drug_type and row["x_id"] in drug_map:
            src = drug_map[row["x_id"]]
            bucket = drug_nbrs.setdefault(src, [])
            tgt_map = entity_idx.get(row["y_type"], {})
            if len(bucket) < max_neighbors and row["y_id"] in tgt_map:
                bucket.append((src, rid, tgt_map[row["y_id"]]))

        if row["y_type"] == disease_type and row["y_id"] in disease_map:
            dst = disease_map[row["y_id"]]
            bucket = disease_nbrs.setdefault(dst, [])
            src_map = entity_idx.get(row["x_type"], {})
            if len(bucket) < max_neighbors and row["x_id"] in src_map:
                bucket.append((src_map[row["x_id"]], rid, dst))

    return drug_nbrs, disease_nbrs
