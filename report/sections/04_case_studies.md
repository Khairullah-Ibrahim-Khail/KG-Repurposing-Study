# 4. Case Studies (Q4)

> Predicted drugs, scores, and KG paths all come from files in `results/predictions/`. Nothing is typed by hand.

## 4.0 Relationship to the Paper's Case Studies

Huang et al. (2024) run case studies with the **full** TxGNN (512-dim, pre-trained node features). This reproduction deliberately does **not** copy those, because:

1. The full model does not fit in 8 GB VRAM — everything here is `scaled_reproduction` (64-dim).
2. The paper's chosen diseases may not land in the zero-shot test split of our PrimeKG download.
3. The paper's pre-trained features are not public; we use learnable embeddings.

So we run our **own** case studies on the scaled model, restricted to diseases that actually sit in our zero-shot test split with enough labels to score. These are not claimed to mirror the paper.

| Aspect | Paper (Huang et al. 2024) | This reproduction |
|--------|--------------------------|-------------------|
| Model | Full TxGNN, 512-dim, pre-trained features | Scaled TxGNN, 64-dim, learnable embeddings |
| Disease choice | Authors' own | Constrained to our zero-shot test split |
| Case A | Rare disease (authors' pick) | Familial Hypertrophic Cardiomyopathy (n_pos=1) |
| Case B | Well-studied disease (authors' pick) | Staphylococcus Aureus Infection (n_pos=45) |
| Headline | Full model succeeds on rare diseases | Scaled model fails at n_pos=1; partial at n_pos=45 |

## 4.0b How the Diseases Were Chosen

Both were fixed from the per-disease results in `results/txgnn/zeroshot/seed_42/txgnn.json` **before** any top-K list was inspected (only AUPRC and n_pos were checked at selection time).

- **Case A (rare):** Familial Hypertrophic Cardiomyopathy (id=24573), n_pos=1. Replaced an earlier pick (Hutchinson-Gilford Progeria) once it turned out Progeria has zero therapeutic edges in every split. Logged in the changelog.
- **Case B (well-studied):** Staphylococcus Aureus Infection (id=5545), n_pos=45. Replaced Type 2 Diabetes (id 5148 was absent from the zero-shot seed-42 test split). Both come from zero-shot because only that run stores per-disease `top_k_drugs`.

## 4.1 Files

- Model: TxGNN (`scaled_reproduction`), zero-shot split, seed=42
- `results/predictions/case_study_caseA_txgnn.csv`, `case_study_caseB_txgnn.csv`
- `results/predictions/case_study_caseA_paths_txgnn.csv`, `case_study_caseB_paths_txgnn.csv`
- Regenerate: `python -m repurpose.cases.case_runner`

## 4.2 Case A — Familial Hypertrophic Cardiomyopathy (id=24573, n_pos=1)

**Background.** FHC is an inherited thickening of the heart muscle driven by sarcomere-gene mutations (MYH7, MYBPC3). Its only PrimeKG indication edge is Propranolol (DB00571), a non-selective beta-blocker.

**Top-10 indication predictions (`case_study_caseA_txgnn.csv`, seed=42):**

| Rank | Drug | DrugBank ID | Score | Positive |
|------|------|-------------|-------|---------|
| 1 | Lutetium Lu 177 dotatate | DB13985 | -1.038 | No |
| 2 | Lactose | DB04465 | -1.206 | No |
| 3 | Emapalumab | DB14724 | -1.226 | No |
| 4 | Cysteine | DB00151 | -1.234 | No |
| 5 | Sulfapyridine | DB00891 | -1.251 | No |
| 6 | Etoposide | DB00773 | -1.263 | No |
| 7 | Oleandomycin | DB11442 | -1.265 | No |
| 8 | Etretinate | DB00926 | -1.268 | No |
| 9 | Quinacrine | DB01103 | -1.283 | No |
| 10 | Umifenovir | DB13609 | -1.293 | No |

All scores negative; no positive appears in the top-20; Propranolol is absent.

**Per-disease AUPRC (indication, seed=42): 0.025.** The single positive drug ranks poorly.

**KG paths (`case_study_caseA_paths_txgnn.csv`).** The three drugs traced (Milrinone DB00235, Amrinone DB01427, Dipyridamole DB00975) are FHC **contraindications**, not indications. Their 2-hop paths share a shape:

```
Drug → [contraindication] → hypertrophic cardiomyopathy → [disease_disease] → familial hypertrophic cardiomyopathy
Drug → [drug_drug] → Propranolol → [indication] → familial hypertrophic cardiomyopathy
```

The model reaches FHC mainly through the broader "hypertrophic cardiomyopathy" node — disease-similarity propagation, not drug-specific routes.

**Clinical note.** The top picks (a cancer radiopharmaceutical; an excipient) are implausible for FHC, consistent with AUPRC 0.025 and a 64-dim model with one known drug to work from.

## 4.3 Case B — Staphylococcus Aureus Infection (id=5545, n_pos=45)

**Background.** A common gram-positive pathogen behind skin infections, pneumonia, endocarditis, and sepsis. Many antibiotic classes apply; MRSA needs mupirocin, daptomycin, or vancomycin.

**Top-10 indication predictions (`case_study_caseB_txgnn.csv`, seed=42):**

| Rank | Drug | DrugBank ID | Score | Positive |
|------|------|-------------|-------|---------|
| 1 | Sulfapyridine | DB00891 | 4.783 | No |
| 2 | Etoposide | DB00773 | 4.708 | No |
| 3 | Mupirocin | DB00410 | 4.686 | No |
| 4 | Oleandomycin | DB11442 | 4.664 | No |
| 5 | Doxycycline | DB00254 | 4.659 | No |
| 6 | Carboplatin | DB00958 | 4.590 | No |
| 7 | Emapalumab | DB14724 | 4.558 | No |
| 8 | Bleomycin | DB00290 | 4.542 | No |
| 9 | Phenylbutyric acid | DB06819 | 4.537 | No |
| 10 | Minocycline | DB01017 | 4.524 | No |
| 18 | Benzylpenicillin | DB01053 | 4.366 | **Yes** |

First positive at rank 18 (Benzylpenicillin / Penicillin G). Three of the top-10 are cancer drugs.

**Per-disease AUPRC (indication, seed=42): 0.088.**

**Plausible top-20 picks:** Mupirocin (3, MRSA decolonization), Doxycycline (5), Minocycline (10), Oleandomycin (4, old macrolide), Benzylpenicillin (18, first-line MSSA).
**Implausible picks:** Etoposide (2), Carboplatin (6), Bleomycin (8) — antineoplastics with no role here.

**KG paths (`case_study_caseB_paths_txgnn.csv`).** The three approved drugs traced (Cefprozil DB01150, Cefdinir DB00535, Tazobactam DB01606) follow:

```
Drug → [indication] → staphylococcal infection → [disease_disease] → staphylococcus aureus infection
Drug → [drug_drug] → Ceftazidime → [indication] → staphylococcus aureus infection
```

Again the dominant route is through a related disease node plus disease-disease similarity — the same mechanism as Case A.

**Discrepancy.** The cancer drugs likely share KG neighbors (immune-related nodes) with the antibiotics; at 64 dims with no pre-trained features, the model can't separate mechanism of action.

## 4.4 Interpretation Notes

- Both cases use the 64-dim scaled model; rankings are not clinical recommendations.
- Disease-disease similarity dominates both, matching the Q6 finding that the affinity head is load-bearing.
- Case A (n_pos=1): the rare-disease zero-shot claim is not confirmed at this scale.
- Case B (n_pos=45): several plausible antibiotics surface, but embedding noise puts cancer drugs in the top-10.
- These observations are specific to seed=42, zero-shot, scaled model — not claims about the published TxGNN.
