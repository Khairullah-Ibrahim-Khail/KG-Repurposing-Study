"""Ranking metrics for drug-disease link prediction."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (primary metric)."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve (secondary metric)."""
    return float(roc_auc_score(y_true, y_score))


def hits_at_k(scores: np.ndarray, positives: np.ndarray, k: int = 10) -> float:
    """Fraction of a disease's positive drugs that land in its top-k ranking."""
    n_pos = positives.sum()
    if n_pos == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(positives[top].sum() / n_pos)


def flat_scores(y_true, y_score, task: str) -> dict:
    """Pooled evaluation over all pairs; matches the result-JSON metric block."""
    return {
        "task": task,
        "n_pairs": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auprc": pr_auc(y_true, y_score),
        "auroc": roc_auc(y_true, y_score),
        "prevalence": float(y_true.mean()),
    }
