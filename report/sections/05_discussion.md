# 5. Discussion

## 5.0 Claim-by-Claim Reconciliation

Each question was posed with an expected answer. Here is what the data actually says.

---

### Claim 1 — "Transformers / LLMs do better when trained with a knowledge graph"

**Status: not proven; the claim misnames the model class.**

TxGNN is a GNN (HGT). The "Transformer" in HGT is its attention aggregation, not a text model. No language model is trained here. The testable version is: *does GNN message passing over the KG beat a no-message-passing baseline?*

The data:
- Standard, indication AUPRC: no-KG **wins** (0.893 vs 0.825) — message passing hurts.
- Zero-shot, indication AUPRC: KG marginally ahead (0.714 vs 0.704, within noise).
- Zero-shot, contraindication AUPRC: KG clearly ahead (0.831 vs 0.725).

**Conclusion:** GNN+KG does not consistently beat no-KG at this scale. The benefit is task- and split-specific. "KG always helps" is **not supported**.

---

### Claim 2 — "There is a better alternative to two-phase training"

**Status: not found.**

Both alternatives lose on the pre-set criterion (zero-shot indication AUPRC at equal-or-lower compute): two-phase 0.736 @ 71.6s, single-stage 0.706 @ 130.7s, joint-contrastive 0.670 @ 97.1s. Two-phase is best **and** fastest. The question is answered in the negative.

---

### Claim 3 — "Zero-shot prediction is preferred for TxGNN"

**Status: confirmed, with a concrete reason.**

92% of PrimeKG diseases have no indication edge. A supervised per-disease classifier has no labels for them. Zero-shot is not a stylistic choice — it is the only workable option for rare and unlabelled diseases. The degradation curve backs this up: standard-split AUPRC collapses as training edges vanish, while zero-shot TxGNN holds ~0.74 across all held-out diseases.

---

### Claim 4 — "The paper has two case studies"

**Status: true, but not replicated here.**

The paper's cases use the full 512-dim TxGNN with pre-trained features — neither available to us. We run original cases on the scaled model: Case A (FHC, n_pos=1) **fails** (Propranolol absent from top-20, AUPRC 0.025); Case B (S. aureus, n_pos=45) is a **partial** success (Benzylpenicillin rank 18, three cancer drugs in the top-10, AUPRC 0.088). This neither confirms nor contradicts the paper — it's a different model scale.

---

### Claim 5 — "Comparative analysis of GNN and TxGNN in tabular form"

**Status: delivered (Section 3.5).**

The table spans both splits, seeds [42, 0, 1], indication and contraindication AUPRC/AUROC, wall-clock, and a `paper_reported` column from Suppl. S1–S2. Generated from `results/` — no hand-typed numbers. Headline: on the standard split the simplest model (gnn_no_kg, 0.7s) has the highest indication AUPRC; TxGNN's edge appears only zero-shot, and only by 3.2 points.

---

### Claim 6 — "Attention augmentation in TxGNN is optional"

**Status: wrong direction — it's detrimental.**

delta = AUPRC(attn=ON) − AUPRC(attn=OFF) = −0.036, past the −0.02 "harmful" threshold. Mean aggregation lifts zero-shot indication AUPRC from 0.736 to 0.772. "Optional" understates it: the right call is **don't use learnable HGT attention in this setup**. This agrees with the paper's own choice — Huang et al. found learnable attention ineffective and used a fixed degree-based gate (λ=0.7) instead.

---

## 5.1 KG Augmentation Doesn't Consistently Help (Q1)

On PrimeKG, standard split, the no-KG embedding model beats the 2-layer HGT on indication AUPRC (0.893 ± 0.006 vs 0.825 ± 0.022). Zero-shot indication is within noise (0.704 vs 0.714); KG helps zero-shot contraindication (0.831 vs 0.725).

This is scoped to: PrimeKG (2026-06-24), scaled reproduction (64-dim, 2 layers, ~2.4M edges), random-negative evaluation, seeds [42, 0, 1]. Possible reasons: 2-layer HGT over non-therapeutic edges adds noise; 64 dims may be too little capacity to use the neighborhood; the 512-dim model likely has enough. Do not generalize beyond this setup.

## 5.2 Two-Phase Stays Best for Zero-Shot (Q2)

Neither alternative beats two-phase on zero-shot indication under the pre-set criterion, and both are slower. The pretraining phase appears to position embeddings so the later fine-tune generalizes better to unseen diseases; collapsing the phases loses that. Joint-contrastive's higher zero-shot contraindication (0.854) comes with lower, noisier indication — worth noting, not winning.

## 5.3 Coverage Forces Zero-Shot (Q3)

92% of PrimeKG diseases lack an indication edge; on our data 9,388 indication pairs cover 17,080 diseases. A standard supervised model can't touch the other ~92%. The degradation curve confirms the collapse toward the zero-edge regime, while zero-shot TxGNN holds ~0.74. This is a constraint, not a preference.

## 5.4 Case-Study Observations (Q4)

FHC (n_pos=1): the model fails to rank Propranolol in the top-20; reasoning runs through disease-disease similarity. S. aureus (n_pos=45): several plausible antibiotics appear in the top-20, but three antineoplastics show up in the top-10 — embedding noise at 64 dims. Both lean on the similarity pathway.

## 5.5 Attention Is Detrimental at This Scale (Q6)

Removing HGT attention raises zero-shot indication AUPRC from 0.736 to 0.772 (delta −0.036, past the 0.02 threshold). The affinity head — not attention — is what matters: removing it drops AUPRC to 0.726. This is independently supported by the paper, which replaced learnable attention with a fixed degree-based gate (λ=0.7). A likely mechanism: HGT attention has O(heads × dim²) parameters per layer; at 64 dims with limited data, those weights overfit, while parameter-free mean aggregation is more stable.

Scope: scaled reproduction, RTX 4060 8 GB, hidden_dim=64, 2 layers, PrimeKG, seeds [42, 0, 1]. The paper's gating choice is convergent evidence, not proof.

## 5.6 Limits of the Scaled Reproduction

Four compounding deviations apply throughout: hidden_dim 512→64, layers 3→2, heads 8→4, learnable embeddings instead of pre-trained features. A 64-dim model has ~8× less representational capacity than the published one, so our numbers are not comparable to the paper's. Paper figures stay in `paper_reported`; ours stay in `scaled_reproduction`; the two are never merged.

## 5.7 Evaluation-Protocol Note

The Q5 table uses random-negative AUPRC (1:5) for every model, for consistency. TxGNN result files also carry a per-disease full-ranking AUPRC (much lower in absolute terms because of the 1:7957 imbalance when all drugs are candidates); that metric drives the Q3 degradation curve and the case-study AUPRCs, while the random-negative metric drives all cross-model comparisons.
