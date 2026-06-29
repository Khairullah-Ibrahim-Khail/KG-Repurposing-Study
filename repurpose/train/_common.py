"""Shared helpers for the training loops — pair construction, random-negative
flat evaluation, and the common result-JSON envelope.
"""

from datetime import date

import numpy as np
import torch

from repurpose.eval.scoring import pr_auc, roc_auc

TREATMENT_RELATIONS = frozenset({"indication", "contraindication"})


def pair_tensors(df, drug_map, disease_map, relations=TREATMENT_RELATIONS):
    """``relation -> (drug_idx, disease_idx)`` long tensors for valid rows."""
    out = {}
    for rel, grp in df.groupby("relation"):
        if relations is not None and rel not in relations:
            continue
        pairs = [
            (drug_map[a], disease_map[b])
            for a, b in zip(grp["x_id"], grp["y_id"])
            if a in drug_map and b in disease_map
        ]
        if not pairs:
            continue
        out[rel] = (
            torch.tensor([p[0] for p in pairs], dtype=torch.long),
            torch.tensor([p[1] for p in pairs], dtype=torch.long),
        )
    return out


def _neg_scores(h, drug_idx, n_diseases, drug_type, disease_type, device, ratio):
    neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * ratio,), device=device)
    neg_drug = drug_idx.repeat(ratio)
    return (h[drug_type][neg_drug] * h[disease_type][neg_dis]).sum(-1)


def val_auprc(h, pairs, n_diseases, drug_type, disease_type, device, ratio=5) -> float:
    """Mean random-negative AUPRC across relations — used for early stopping."""
    scores = []
    for _, (d, dis) in pairs.items():
        d, dis = d.to(device), dis.to(device)
        pos = (h[drug_type][d] * h[disease_type][dis]).sum(-1).cpu().numpy()
        neg = _neg_scores(h, d, n_diseases, drug_type, disease_type, device, ratio).cpu().numpy()
        yt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        ys = np.concatenate([pos, neg])
        scores.append(pr_auc(yt, ys))
    return float(np.mean(scores)) if scores else 0.0


def flat_metrics(h, pairs, n_diseases, drug_type, disease_type, device, ratio=5) -> dict:
    """Random-negative AUPRC/AUROC per relation (the cross-model protocol)."""
    out = {}
    for rel, (d, dis) in pairs.items():
        d, dis = d.to(device), dis.to(device)
        pos = (h[drug_type][d] * h[disease_type][dis]).sum(-1).cpu().numpy()
        neg = _neg_scores(h, d, n_diseases, drug_type, disease_type, device, ratio).cpu().numpy()
        yt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        ys = np.concatenate([pos, neg])
        out[rel] = {"auprc": pr_auc(yt, ys), "auroc": roc_auc(yt, ys), "n_pairs": len(yt)}
    return out


def envelope(model_name, reproduction_type, split, seed, wall_s) -> dict:
    """The shared header block every result JSON starts with."""
    cuda = torch.cuda.is_available()
    return {
        "model": model_name,
        "reproduction_type": reproduction_type,
        "split": split,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0,
        "date": date.today().isoformat(),
        "wall_clock_seconds": wall_s,
    }
