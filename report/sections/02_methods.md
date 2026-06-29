# 2. Methods

## 2.1 Data and Splits

PrimeKG (Chandak et al., 2023), downloaded from Harvard Dataverse (dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM): 8,100,498 edges, license CC0.

**Standard split** — train/val/test over (drug, disease) pairs; a disease can appear in all three sets.
**Zero-shot split** — a set of diseases is withheld from training entirely (641 held-out diseases, none with an approved therapy in train).

The audit in `scripts/check_leakage.py` confirms 0 leaking diseases for seeds [42, 0, 1] (`results/metrics/leakage_check_seed{n}.json`).

Two relation families are kept out of message passing — `anatomy_protein_present` (3.03M) and `drug_drug` (2.67M) — leaving ~2.4M of 8.1M edges. This is a documented VRAM-driven deviation.

## 2.2 Models

### Plain GNN — KG condition (Q1)
Two-layer HGT encoder. Node features are a learnable `nn.Embedding` per type (Xavier init); hidden dim 64; dot-product scoring. Phase 1 is link-prediction pretraining (30 epochs); Phase 2 fine-tunes on therapeutic edges (≤100 epochs, early-stop patience 10).

### Plain GNN — no-KG condition (Q1)
The same model with `depth=0`: the encoder returns embeddings with no aggregation. Both conditions share Phase 1, so "no-KG" means no message passing, not the absence of KG-informed pretraining.

### Scaled TxGNN
The same HGT encoder plus a `DiseaseAffinityHead` (cosine projection, k=5 nearest support diseases). Two-phase training: link-prediction pretrain, then therapeutic task with a triplet metric loss (`sim_loss_weight=0.3`); attention and affinity both on.

Deviations from the paper: hidden_dim 512→64, layers 3→2, heads 8→4, learnable embeddings instead of the (unavailable) pre-trained features.

### Alternatives (Q2)
- **JointTaskNet** — KG link-prediction and therapeutic loss together from epoch 1; no phase split.
- **ContrastiveNet** — InfoNCE disease similarity and therapeutic loss jointly.

### Ablations (Q6)
- **attn=ON** — full HGT attention.
- **attn=OFF** — SAGEConv mean aggregation in place of attention.

## 2.3 Evaluation

**Primary:** AUPRC. **Secondary:** AUROC. **Protocol:** random negatives at 1:5 for all flat evaluations; per-disease full-ranking for TxGNN zero-shot. **Seeds:** [42, 0, 1], reported as mean ± std. Tables come from `scripts/make_tables.py` reading the result JSONs — never typed by hand.

## 2.4 Leakage Check

Zero-shot results are accepted only after the audit passes: no held-out test disease may carry a training treatment edge. Output: `results/metrics/leakage_check_seed{n}.json`; status PASS for every seed, verified before any zero-shot run.
