"""Train the plain GNN baseline (Q1 / Q5).

    python scripts/train_gnn.py --split both --seeds 42 0 1
    python scripts/train_gnn.py --depth 0 --model_name gnn_no_kg   # no-KG control

Writes results/gnn/<split>/seed_<n>/<model_name>.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Read the CSV before any torch/PyG import (allocator safety).
from repurpose.graph.kg_ingest import load_graph
from repurpose.graph.split_builder import make_all_splits

RESULTS = ROOT / "results" / "gnn"
SPLITS = ROOT / "data" / "splits"


def splits_ready(seed: int) -> bool:
    return all((SPLITS / s / f"seed_{seed}" / f"{p}.csv").exists()
               for s in ("standard", "zeroshot") for p in ("train", "val", "test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2, help="0 = no message passing (no-KG)")
    ap.add_argument("--model_name", default="gnn_baseline")
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print("[setup] loading PrimeKG ...")
    edges, entity_idx, graph = load_graph()
    for seed in args.seeds:
        if not splits_ready(seed):
            make_all_splits(edges, seeds=[seed])

    import torch
    from repurpose.util.device import require_cuda
    from repurpose.nets.plain_gnn import PlainHGT
    from repurpose.train.phase1_pretrain import run_pretrain
    from repurpose.train.phase2_finetune import run_finetune

    device = require_cuda(0)
    node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
    targets = ["standard", "zeroshot"] if args.split == "both" else [args.split]

    for seed in args.seeds:
        for split in targets:
            print(f"\n{'='*60}\n[run] {args.model_name} | {split} | seed={seed}\n{'='*60}")
            torch.manual_seed(seed)
            model = PlainHGT(graph.metadata(), node_counts,
                             hidden_dim=args.hidden_dim, depth=args.depth)
            run_pretrain(model, graph, device, epochs=args.pretrain_epochs, lr=args.lr,
                         results_dir=RESULTS, model_name=args.model_name, seed=seed)
            result = run_finetune(model, graph, entity_idx,
                                  SPLITS / split / f"seed_{seed}", device,
                                  epochs=args.finetune_epochs, lr=args.lr,
                                  results_dir=RESULTS, model_name=args.model_name,
                                  split_name=split, seed=seed)
            print(json.dumps(result, indent=2))
    print(f"\n[done] {RESULTS}")


if __name__ == "__main__":
    main()
