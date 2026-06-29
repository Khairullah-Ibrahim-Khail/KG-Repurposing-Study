"""repurpose — scaled TxGNN drug-repurposing study on PrimeKG.

Top-level package. Sub-packages:
    graph  — PrimeKG ingestion, train/val/test partitioning, leakage auditing
    nets   — model definitions (plain GNN, scaled TxGNN, ablations, alternatives)
    train  — training loops (two-phase and joint variants)
    eval   — scoring metrics and per-disease zero-shot evaluation
    cases  — Q4 case-study extraction
    util   — device setup and run logging
"""

__all__ = ["graph", "nets", "train", "eval", "cases", "util"]
