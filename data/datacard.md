# Data Card — PrimeKG

## Source
PrimeKG (Chandak, Huang, Zitnik, 2023), Harvard Dataverse,
doi:10.7910/DVN/IXA7BM. License: CC0 (public domain).

Files used: `kg.csv` (edge list, source of truth), `nodes.tab`,
`disease_features.tab`, `drug_features.tab`. Fetch with
`python scripts/fetch_primekg.py`.

## Shape (from our download, not from the paper)
- Edges: 8,100,498
- Node types: 10 (disease, drug, gene/protein, biological_process, …)
- Therapeutic edges: 18,776 indication, 61,350 contraindication
- Diseases with ≥1 treatment edge: 4,005 of 17,080 (so ~76% are drug-naive)

Column names: `relation, display_relation, x_index, x_id, x_type, x_name,
x_source, y_index, y_id, y_type, y_name, y_source`. Therapeutic direction is
x=drug, y=disease.

## Splits
Built by `repurpose/graph/split_builder.py` for seeds [42, 0, 1].
- **standard** — random edge hold-out (10% test, 10% val); diseases may recur.
- **zeroshot** — 20% of diseases withheld from training entirely; all their
  therapeutic edges go to val/test. 641 held-out test diseases.

The zero-shot regime is the one where the leakage audit matters.

## Message-passing graph
`anatomy_protein_present` (3.03M) and `drug_drug` (2.67M) are dropped from
aggregation for VRAM reasons, leaving ~2.4M edges. They remain in `kg.csv`.

## Known limitations
- Scaled reproduction: learnable embeddings, not the paper's pre-trained features.
- Edge filtering means the model never sees ~70% of edges during aggregation.
- Per-disease full-ranking AUPRC is tiny in absolute terms (1:7957 imbalance);
  cross-model comparisons use the random-negative (1:5) protocol instead.
