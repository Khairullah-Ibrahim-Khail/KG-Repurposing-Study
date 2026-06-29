# Literature Notes

Primary sources only. Background papers (KG-BERT, DRAGON) are context for the
LM+KG direction — they are **not** reproduced and their numbers (different
datasets) do not transfer to PrimeKG drug repurposing.

## TxGNN — Huang et al. (2024), Nature Medicine
DOI 10.1038/s41591-024-03233-x · PMC11326339 (preprint), PMC11645266 (final).
The paper being reproduced. Takeaways used here:
- An HGT encoder over PrimeKG plus a disease-similarity module for zero-shot transfer.
- Two-phase training: self-supervised KG pretraining, then therapeutic fine-tune.
- Suppl. S1/S2 figures (mean ± std): standard indication 0.91 ± 0.02, contraindication
  0.82 ± 0.01; zero-shot indication 0.90 ± 0.02, contraindication 0.80 ± 0.01.
- HAN standard contraindication 0.84 ± 0.00 (beats TxGNN there).
- ~92% of PrimeKG diseases have no indication edge (main text).
- A learnable attention gate was found ineffective and replaced by a fixed
  degree-based gate (λ=0.7) (Methods).
- The abstract's +49.2% / +35.1% headline is not tied to a split; not used.

## PrimeKG — Chandak et al. (2023), Scientific Data
DOI 10.1038/s41597-023-01960-3. The knowledge graph. Used for the data card
(node/edge counts come from our own download, not the paper).

## KG-BERT — Yao et al. (2019), arXiv:1909.03193
Background. Treats KG triples as text and classifies validity. Numbers not
confirmed from PDF; context only.

## DRAGON — Yasunaga et al. (2022), NeurIPS, arXiv:2210.09338
Background. Joint LM+KG pretraining; ~+5% on QA tasks (ConceptNet/UMLS). Does not
transfer to PrimeKG repurposing.

## Claims register
- Q1/Q5 paper figures → TxGNN Suppl. S1/S2 (above).
- Q3 coverage (92%) → TxGNN main text.
- Q6 gating choice → TxGNN Methods.
- All `scaled_reproduction` numbers → files under `results/`.
