# CHANGELOG — build log and deviation tracker

Append-only. One block per working session. Records what was done, what was
assumed vs. verified, and any deviation from the ground rules in `GUIDE.md`.

---

## Session 1 — project scaffold

**Done:** laid out the package (`repurpose/`), scripts, configs, and the result
schema; wrote `GUIDE.md`; confirmed the GPU (RTX 4060, 8 GB).

**Verified:** GPU name/VRAM via `nvidia-smi`.

**Deviation:** none — no numbers produced yet.

---

## Session 2 — data + pipeline

**Done:** PrimeKG ingest (`graph/kg_ingest.py`), split builder
(`graph/split_builder.py`), leakage audit (`graph/leakage_audit.py`); model and
training code; first GNN baseline runs.

**Verified (from the data):**
- kg.csv: 8,100,498 edges (not the ~4.05M sometimes quoted from summaries).
- Indication edges 18,776; contraindication 61,350.
- 4,005 of 17,080 diseases carry a treatment edge → 13,075 are drug-naive.
- Drug→disease direction confirmed (x_type=drug, y_type=disease).
- Leakage audit PASS for seeds 42/0/1 (0 leaking diseases).

**Deviations:**
- hidden_dim set to 64 (512 OOMs; even 128 OOMs on backward).
- Node features are learnable embeddings (paper's pre-trained features not public).
- `anatomy_protein_present` and `drug_drug` excluded from message passing
  (8.1M → ~2.4M edges).

**Surprise:** 76% of diseases have zero approved drugs — a larger zero-shot pool
than expected.

---

## Session 3 — baselines + TxGNN

**Verified (result files):**
- GNN with KG (random-negative AUPRC), standard indication mean ≈ 0.824.
- GNN no-KG (depth=0) standard indication mean ≈ 0.893 — **beats** the KG model,
  and trains ~30× faster. Reported as-is per rule 2.
- Zero-shot indication is within noise between the two.

**Surprise:** the no-message-passing model wins on the standard split. The "no-KG"
label means no aggregation in the forward pass — both still use KG edges during
Phase 1 pretraining.

---

## Session 4 — alternatives + ablations

**Verified (result files):**
- Two-phase zero-shot indication 0.736 ± 0.011 (71.6s) beats single-stage
  0.706 ± 0.033 (130.7s) and joint-contrastive 0.670 ± 0.038 (97.1s).
- Q6 zero-shot: txgnn 0.736, no_attn 0.772, no_sim 0.726, no_both 0.762.
  delta = −0.036 → attention **detrimental** (and no_attn runs ~22× faster).

**Surprise:** attention hurts; single-stage is slower than two-phase.

---

## Session 5 — case studies + report

**Done:** Q4 cases (FHC id=24573 n_pos=1; S. aureus id=5545 n_pos=45) from
zeroshot/seed_42; generated comparison and ablation tables; wrote the five report
sections and built the web view.

**Deviation (logged):** Case A switched from Hutchinson-Gilford Progeria (no
therapeutic edges anywhere) to FHC; Case B switched from Type 2 Diabetes (absent
from the zero-shot seed-42 test split) to S. aureus. Both confirmed from
per-disease results before any top-K list was inspected.

**Verified:** Case A — Propranolol not in top-20, AUPRC 0.025. Case B —
Benzylpenicillin rank 18, three cancer drugs in the top-10, AUPRC 0.088.

<!-- new sessions below -->
