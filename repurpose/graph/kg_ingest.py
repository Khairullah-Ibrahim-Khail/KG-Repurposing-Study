"""Read PrimeKG from disk and turn it into a PyG heterogeneous graph.

Expects ``data/raw/kg.csv`` to be present (fetch it with
``scripts/fetch_primekg.py`` first).

Why torch is imported lazily: importing a PyG/torch C-extension before pandas
parses a large CSV can corrupt the process allocator on Windows and crash the
read. To stay safe everywhere, this module keeps the torch import *inside* the
function that needs it, and callers read the CSV before any model import.
"""

from pathlib import Path

import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PKG_ROOT / "data" / "raw"
PROCESSED_DIR = PKG_ROOT / "data" / "processed"

# Drug<->disease relations that constitute the prediction task.
TREATMENT_RELATIONS = frozenset({"indication", "contraindication"})

# Two relation families are far too large to message-pass through on an 8 GB
# card and neither sits on a direct drug->disease repurposing route, so they
# are dropped from the graph used for aggregation:
#   anatomy_protein_present  ~3.04M edges
#   drug_drug                ~2.67M edges
# Dropping them takes the graph from ~8.1M to ~2.4M edges. This is recorded as
# a scaled-reproduction deviation.
OVERSIZED_RELATIONS = frozenset({"anatomy_protein_present", "drug_drug"})


def read_edge_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load ``kg.csv`` into a DataFrame."""
    csv_path = raw_dir / "kg.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Fetch it first: python scripts/fetch_primekg.py"
        )
    edges = pd.read_csv(csv_path, low_memory=False)
    print(f"[ingest] {len(edges):,} edges loaded; columns={list(edges.columns)}")
    return edges


def index_entities(edges: pd.DataFrame) -> dict:
    """Map every node to a contiguous local integer index, grouped by type.

    Returns ``{node_type: {raw_id: local_index}}`` built from both the ``x``
    (head) and ``y`` (tail) sides of the edge table.
    """
    by_type: dict = {}
    for type_col, id_col in (("x_type", "x_id"), ("y_type", "y_id")):
        for ntype, chunk in edges.groupby(type_col):
            table = by_type.setdefault(ntype, {})
            for raw_id in chunk[id_col].unique():
                if raw_id not in table:
                    table[raw_id] = len(table)
    return by_type


def to_hetero_graph(
    edges: pd.DataFrame,
    entity_idx: dict,
    drop_relations: frozenset = OVERSIZED_RELATIONS,
):
    """Assemble a ``HeteroData`` object from the edge table.

    Each distinct ``(x_type, relation, y_type)`` triple becomes its own edge
    store. Relations listed in ``drop_relations`` are skipped to respect the
    VRAM budget.
    """
    import torch  # deferred on purpose — see module docstring
    from torch_geometric.data import HeteroData  # deferred for the same reason

    graph = HeteroData()

    # Identity node ids; learned features live in the model's embedding tables.
    for ntype, id_map in entity_idx.items():
        count = len(id_map)
        graph[ntype].num_nodes = count
        graph[ntype].node_id = torch.arange(count)

    dropped = set()
    grouped = edges.groupby(["x_type", "relation", "y_type"])
    for (x_type, relation, y_type), chunk in grouped:
        if relation in drop_relations:
            dropped.add(relation)
            continue
        heads = torch.tensor(
            [entity_idx[x_type][i] for i in chunk["x_id"]], dtype=torch.long
        )
        tails = torch.tensor(
            [entity_idx[y_type][i] for i in chunk["y_id"]], dtype=torch.long
        )
        graph[x_type, relation, y_type].edge_index = torch.stack([heads, tails], dim=0)

    if dropped:
        names = ", ".join(sorted(dropped))
        print(f"[ingest] dropped {len(dropped)} oversized relation(s): {names}")

    return graph


def therapeutic_subset(edges: pd.DataFrame) -> pd.DataFrame:
    """Return only the indication/contraindication rows."""
    return edges[edges["relation"].isin(TREATMENT_RELATIONS)].copy()


def load_graph(raw_dir: Path = RAW_DIR):
    """Convenience entry point: ``(edges_df, entity_index, hetero_graph)``."""
    edges = read_edge_table(raw_dir)
    entity_idx = index_entities(edges)
    graph = to_hetero_graph(edges, entity_idx)
    print(f"[ingest] node types: {list(entity_idx.keys())}")
    print(f"[ingest] {len(entity_idx)} node types, {len(graph.edge_types)} edge types")
    return edges, entity_idx, graph
