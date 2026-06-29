# GUIDE — project rules and conventions

Read this before changing code, writing a number, or making a claim. These are
hard constraints.

## 0. Hardware

NVIDIA GeForce RTX 4060, 8 GB VRAM. Train on CUDA only — `repurpose.util.device.
require_cuda()` aborts if no GPU is present rather than falling back to CPU.
Because VRAM is tight, models are sized down and every result file is tagged
`scaled_reproduction`.

```python
import torch
assert torch.cuda.is_available(), "GPU required"
device = torch.device("cuda:0")
```

## 1. Non-negotiables

1. **Traceability.** Every metric in the report or on the site maps to a file in
   `results/`. If it isn't in a result file or a cited primary source, it does
   not appear.
2. **Null results count.** If KG augmentation loses to no-KG, the report says so.
   Run the experiment as designed; record what happens.
3. **No universal claims.** Scope to dataset, split, seeds, and variant.
4. **Reproduction ≠ original.** `scaled_reproduction` is our number;
   `paper_reported` is Huang et al.'s. Separate columns, never merged.
5. **Leakage is fatal.** A zero-shot result is invalid unless the audit passes
   (no held-out disease has a training treatment edge). Fix the split and re-run
   if it fails.
6. **Primary sources only.** The TxGNN paper, the PrimeKG paper/repo, and the
   official dataset — not blog posts or news.
7. **Plain writing.** Short sentences; say what was done and what happened.

## 2. The six questions → deliverables

- **Q1** plain GNN with vs. without message passing (`scripts/train_gnn.py`,
  `--depth 0` for the control).
- **Q2** two-phase vs. single-stage vs. joint-contrastive
  (`scripts/train_txgnn.py`, `scripts/train_alternatives.py`).
- **Q3** degradation curve (`scripts/degradation_curve.py`).
- **Q4** two case studies (`repurpose.cases.case_runner`).
- **Q5** comparison table (`scripts/make_tables.py`).
- **Q6** attention ablation + decision (`scripts/run_ablations.py`,
  `scripts/ablation_matrix.py`); rule fixed before running:
  |delta| < 0.02 optional, ≥ 0.02 helps, ≤ −0.02 hurts.

## 3. Result-file schema

`results/<group>/<split>/seed_<n>/<model>.json` carries at least:

```json
{
  "model": "...", "reproduction_type": "scaled_reproduction",
  "split": "zeroshot", "seed": 42, "gpu": "...", "vram_gb": 8,
  "date": "YYYY-MM-DD", "wall_clock_seconds": 0,
  "indication": {"auprc": 0.0, "auroc": 0.0},
  "contraindication": {"auprc": 0.0, "auroc": 0.0},
  "notes": ""
}
```

`reproduction_type` is one of `scaled_reproduction`, `paper_baseline`,
`original_ablation`.

## 4. Deviations from the published TxGNN

hidden_dim 512→64, layers 3→2, heads 8→4, learnable embeddings instead of
pre-trained features, and `anatomy_protein_present` + `drug_drug` edges excluded
from message passing (VRAM). All recorded in `CHANGELOG.md`.
