"""Q2 — train the two alternatives to two-phase training.

    python scripts/train_alternatives.py --method both --split both

single_stage      -> results/alt_single_stage/...
joint_contrastive -> results/alt_joint_contrastive/...
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repurpose.graph.kg_ingest import load_graph
from repurpose.graph.split_builder import make_all_splits
from repurpose.graph.leakage_audit import audit_split

SPLITS = ROOT / "data" / "splits"


def splits_ready(seed: int) -> bool:
    return all((SPLITS / s / f"seed_{seed}" / f"{p}.csv").exists()
               for s in ("standard", "zeroshot") for p in ("train", "val", "test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="both",
                    choices=["single_stage", "joint_contrastive", "both"])
    ap.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip_leakage_check", action="store_true")
    args = ap.parse_args()

    print("[setup] loading PrimeKG ...")
    edges, entity_idx, graph = load_graph()
    for seed in args.seeds:
        if not splits_ready(seed):
            make_all_splits(edges, seeds=[seed])
    if not args.skip_leakage_check:
        for seed in args.seeds:
            if audit_split(seed)["status"] == "FAIL":
                sys.exit(f"[ABORT] leakage in zeroshot seed={seed}")

    import torch
    from repurpose.util.device import require_cuda
    from repurpose.nets.single_stage_net import JointTaskNet
    from repurpose.nets.contrastive_net import ContrastiveNet
    from repurpose.train.joint_loops import run_single_stage, run_contrastive

    device = require_cuda(0)
    node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
    meta = graph.metadata()
    methods = ["single_stage", "joint_contrastive"] if args.method == "both" else [args.method]
    targets = ["standard", "zeroshot"] if args.split == "both" else [args.split]

    for method in methods:
        results_dir = ROOT / "results" / f"alt_{method}"
        for seed in args.seeds:
            for split in targets:
                print(f"\n{'='*60}\n[run] {method} | {split} | seed={seed}")
                torch.manual_seed(seed)
                split_dir = SPLITS / split / f"seed_{seed}"
                if method == "single_stage":
                    model = JointTaskNet(meta, node_counts, hidden_dim=args.hidden_dim)
                    result = run_single_stage(model, graph, entity_idx, split_dir, device,
                                              epochs=args.epochs, lr=args.lr,
                                              results_dir=results_dir, model_name=method,
                                              split_name=split, seed=seed)
                else:
                    model = ContrastiveNet(meta, node_counts, hidden_dim=args.hidden_dim)
                    result = run_contrastive(model, graph, entity_idx, split_dir, device,
                                             epochs=args.epochs, lr=args.lr,
                                             results_dir=results_dir, model_name=method,
                                             split_name=split, seed=seed)
                print(json.dumps({k: v for k, v in result.items()
                                  if k != "per_disease_results"}, indent=2))
    print("\n[done] alternatives complete.")


if __name__ == "__main__":
    main()
