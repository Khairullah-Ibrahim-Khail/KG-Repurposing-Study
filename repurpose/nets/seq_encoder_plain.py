"""Q1 no-KG control — same transformer backbone as ``seq_encoder_kg`` but the
input is the entity id embedding only, with no neighbor-triple context. Any
AUPRC gap to the KG variant is attributable to the KG context, not capacity.
"""

import torch
import torch.nn as nn


class PlainSeqEncoder(nn.Module):
    def __init__(self, n_drugs, n_diseases, hidden_dim=128, heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_kg = False

        self.drug_emb = nn.Embedding(n_drugs, hidden_dim)
        self.disease_emb = nn.Embedding(n_diseases, hidden_dim)
        self.drug_token = nn.Parameter(torch.randn(hidden_dim))
        self.disease_token = nn.Parameter(torch.randn(hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, drug_idx, disease_idx, **_ignored):
        drug_seq = (self.drug_emb(drug_idx) + self.drug_token).unsqueeze(1)
        disease_seq = (self.disease_emb(disease_idx) + self.disease_token).unsqueeze(1)
        drug_h = self.encoder(drug_seq)[:, 0, :]
        disease_h = self.encoder(disease_seq)[:, 0, :]
        return self.scorer(torch.cat([drug_h, disease_h], dim=-1)).squeeze(-1)
