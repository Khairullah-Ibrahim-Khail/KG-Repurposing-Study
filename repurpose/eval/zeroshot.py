"""Per-disease zero-shot evaluation and Q3 degradation data.

For every held-out test disease we score all candidate drugs, compute its own
AUPRC/AUROC, and keep the top-K predictions (used by the Q4 case studies).
"""

import numpy as np
import pandas as pd
import torch

from repurpose.eval.scoring import pr_auc, roc_auc

TREATMENT_RELATIONS = frozenset({"indication", "contraindication"})


def per_disease_eval(
    model,
    h: dict,
    test_edges: pd.DataFrame,
    entity_idx: dict,
    device,
    drug_type: str = "drug",
    disease_type: str = "disease",
    top_k: int = 20,
) -> dict:
    """Return ``{"per_disease": [...], "aggregate": {relation: {...}}}``."""
    drug_h = h[drug_type]
    disease_h = h[disease_type]
    drug_map = entity_idx[drug_type]
    disease_map = entity_idx[disease_type]
    n_drugs = drug_h.size(0)
    idx_to_drug = {v: k for k, v in drug_map.items()}

    per_disease = []
    for (disease_id, relation), chunk in test_edges.groupby(["y_id", "relation"]):
        if disease_id not in disease_map:
            continue
        positives = [drug_map[d] for d in set(chunk["x_id"]) if d in drug_map]
        if not positives:
            continue

        dvec = disease_h[disease_map[disease_id]].unsqueeze(0)
        with torch.no_grad():
            scores = (drug_h * dvec).sum(-1).cpu().numpy()

        labels = np.zeros(n_drugs)
        labels[positives] = 1.0
        if labels.sum() in (0, n_drugs):
            continue

        ranked = np.argsort(scores)[::-1][: min(top_k, n_drugs)]
        top = [
            {
                "rank": int(i + 1),
                "drug_id": str(idx_to_drug.get(int(j), f"idx_{j}")),
                "score": float(round(scores[j], 6)),
                "is_positive": bool(labels[j] > 0),
            }
            for i, j in enumerate(ranked)
        ]
        per_disease.append({
            "disease_id": str(disease_id),
            "relation": relation,
            "n_pos": int(labels.sum()),
            "auprc": round(pr_auc(labels, scores), 5),
            "auroc": round(roc_auc(labels, scores), 5),
            "top_k_drugs": top,
        })

    aggregate = {}
    for rel in TREATMENT_RELATIONS:
        rows = [r for r in per_disease if r["relation"] == rel]
        if not rows:
            aggregate[rel] = {"auprc_mean": None, "auprc_std": None,
                              "auroc_mean": None, "auroc_std": None, "n_diseases": 0}
            continue
        aps = [r["auprc"] for r in rows]
        rcs = [r["auroc"] for r in rows]
        aggregate[rel] = {
            "auprc_mean": round(float(np.mean(aps)), 5),
            "auprc_std": round(float(np.std(aps)), 5),
            "auroc_mean": round(float(np.mean(rcs)), 5),
            "auroc_std": round(float(np.std(rcs)), 5),
            "n_diseases": len(rows),
        }

    return {"per_disease": per_disease, "aggregate": aggregate}


def degradation_records(per_disease: list, train_edges: pd.DataFrame) -> list:
    """Pair each disease's zero-shot AUPRC with its training treatment-edge count."""
    counts = (
        train_edges[train_edges["relation"].isin(TREATMENT_RELATIONS)]
        .groupby("y_id").size().to_dict()
    )
    return [
        {
            "disease_id": row["disease_id"],
            "relation": row["relation"],
            "n_train_treatment_edges": int(counts.get(row["disease_id"], 0)),
            "auprc": row["auprc"],
        }
        for row in per_disease
    ]
