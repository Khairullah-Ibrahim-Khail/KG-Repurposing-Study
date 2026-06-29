# 1. Introduction

Repurposing an approved drug for a new disease is faster and cheaper than developing one from scratch. The catch: most computational methods need labelled drug–disease pairs to learn from, and for rare diseases those labels simply do not exist.

TxGNN (Huang et al., 2024) tackles this with a heterogeneous knowledge graph (PrimeKG) and a disease-similarity module, letting it score drugs for diseases that have no approved therapy in the training data. This project takes that system apart and asks six concrete, testable questions about it.

What follows is a **scaled reproduction**. Because of the RTX 4060's 8 GB of VRAM, every model runs at `hidden_dim=64` rather than the published `512`. Each number in this report points back to a file under `results/`; none are entered by hand.

## 1.1 A Note on the Architecture

TxGNN is built on the **Heterogeneous Graph Transformer (HGT)**. The "Transformer" there refers only to HGT's multi-head attention aggregation — it is a **graph neural network**, not a language model and not a text Transformer. We train and evaluate no language models; every result below concerns GNN-style models.

So the question sometimes phrased as "do Transformers/LLMs do better with a knowledge graph?" is restated here as: **does a GNN that message-passes over a KG beat a GNN that does not?** On the standard split the answer is no. Sections 3.1 and 5.1 give the full picture.

## 1.2 The Six Questions (operationalized)

1. Does message passing over the KG improve repurposing over a no-message-passing baseline?
2. Is there a better recipe than TxGNN's two-phase training? (Answer: no — two-phase still wins.)
3. Why is zero-shot prediction the right framing for TxGNN? (Answer: most diseases have no training labels.)
4. What do two case studies — one rare disease, one common — say about the predictions?
5. How does a plain GNN line up against the full TxGNN, head to head? (A generated table.)
6. Is TxGNN's attention necessary for zero-shot performance? (Answer: not just unnecessary — it hurts.)

## 1.3 Data

PrimeKG (Chandak et al., 2023): 8,100,498 edges, 10 node types, 129,375 nodes, obtained from Harvard Dataverse. Therapeutic edges: 18,776 indication, 61,350 contraindication.

## 1.4 Hardware

NVIDIA GeForce RTX 4060, 8 GB VRAM, CUDA throughout. Every output carries the `scaled_reproduction` label.

## 1.5 Reproducibility

The comparison numbers come from `results/metrics/comparison_table.csv` (built by `scripts/make_tables.py`). Leakage audits are in `results/metrics/leakage_check_seed{n}.json` — all PASS.
