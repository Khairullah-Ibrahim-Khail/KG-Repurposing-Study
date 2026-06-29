# KG Drug-Repurposing Study — Scaled TxGNN on PrimeKG

A from-scratch, audit-friendly reproduction of TxGNN (Huang et al., 2024, *Nature
Medicine*) at reduced scale, used to answer six empirical questions about
knowledge-graph augmentation for drug repurposing. Everything runs on a single
8 GB GPU, so all models are `scaled_reproduction` (hidden_dim=64 vs. the
published 512).

## The six questions

| # | Question | Short answer |
|---|----------|--------------|
| Q1 | Does KG message passing beat a no-KG baseline? | No on standard; mixed zero-shot |
| Q2 | Is there a better recipe than two-phase training? | No — two-phase wins and is fastest |
| Q3 | Why zero-shot? | 92% of diseases have no training labels |
| Q4 | What do two case studies show? | Fails on a rare disease; partial on a common one |
| Q5 | GNN vs TxGNN, head to head | Generated table (`results/metrics/comparison_table.csv`) |
| Q6 | Is attention optional? | It is *detrimental* (delta −0.036) |

## Layout

```
repurpose/        Python package
  graph/          PrimeKG ingest, split building, leakage audit
  nets/           plain GNN, scaled TxGNN, ablations, Q2 alternatives, seq encoders
  train/          phase-1 pretrain, phase-2 fine-tunes, joint loops
  eval/           scoring metrics, per-disease zero-shot
  cases/          Q4 case-study extraction
  util/           CUDA device + run log
scripts/          CLI runners + table/figure builders
configs/          YAML hyperparameters
data/             raw PrimeKG, built splits, datacard
results/          per-run JSONs, metrics, predictions, figures
report/sections/  the written report (5 sections)
web/              Vite + React site that reads from results/
```

## Ground rules

1. Every number traces to a file in `results/`; nothing is typed by hand.
2. Null results are reported as-is — no retuning until a hypothesis "wins".
3. Claims are scoped to dataset, split, seeds, and model variant.
4. Reproduction numbers (`scaled_reproduction`) and paper numbers
   (`paper_reported`) are kept in separate columns, never merged.
5. The zero-shot leakage audit must PASS before any zero-shot result is used.

## Quickstart

```bash
python scripts/install_deps.py          # GPU PyTorch + PyG
python scripts/fetch_primekg.py         # download PrimeKG (~1 GB)
python scripts/build_splits.py          # splits + leakage audit
python scripts/train_gnn.py --split both
python scripts/train_gnn.py --depth 0 --model_name gnn_no_kg
python scripts/train_txgnn.py --variant txgnn --split both
python scripts/train_alternatives.py
python scripts/run_ablations.py && python scripts/ablation_matrix.py
python scripts/make_tables.py
python scripts/degradation_curve.py
python -m repurpose.cases.case_runner
```

Then build the site:

```bash
cd web && npm install && npm run build:data && npm run dev
```

## Hardware

NVIDIA GeForce RTX 4060, 8 GB VRAM. `hidden_dim=64` (paper: 512); the message
-passing graph is trimmed to ~2.4M of 8.1M edges to fit VRAM. See `GUIDE.md`.

## Citation

Huang K, Chandak P, Wang Q, et al. "A foundation model for clinician-centered
drug repurposing." *Nature Medicine*, 2024. DOI: 10.1038/s41591-024-03233-x
