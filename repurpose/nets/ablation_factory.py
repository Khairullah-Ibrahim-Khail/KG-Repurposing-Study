"""Q6 ablation builders — thin wrappers that flip the two switches on ScaledTxGNN.

Decision rule (fixed before running): with
``delta = AUPRC(attn=ON) - AUPRC(attn=OFF)`` on the zero-shot split,
``delta < 0.02`` means attention is not load-bearing.
"""

from repurpose.nets.txgnn_scaled import ScaledTxGNN


def build_no_attn(metadata, node_counts, hidden_dim: int = 64, **kw) -> ScaledTxGNN:
    """Attention OFF, affinity ON (Q6 condition B)."""
    return ScaledTxGNN(metadata, node_counts, hidden_dim=hidden_dim,
                       attention=False, affinity=True, **kw)


def build_no_affinity(metadata, node_counts, hidden_dim: int = 64, **kw) -> ScaledTxGNN:
    """Attention ON, affinity OFF — shows the affinity head is load-bearing."""
    return ScaledTxGNN(metadata, node_counts, hidden_dim=hidden_dim,
                       attention=True, affinity=False, **kw)


def build_no_both(metadata, node_counts, hidden_dim: int = 64, **kw) -> ScaledTxGNN:
    """Both OFF — collapses to a plain mean-aggregation GNN."""
    return ScaledTxGNN(metadata, node_counts, hidden_dim=hidden_dim,
                       attention=False, affinity=False, **kw)
