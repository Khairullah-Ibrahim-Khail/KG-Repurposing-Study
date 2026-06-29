"""Train the scaled TxGNN and its Q6 ablation variants.

    python scripts/train_txgnn.py --variant txgnn --split both --seeds 42 0 1
    python scripts/train_txgnn.py --variant txgnn_no_attn       # Q6 condition B
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
VARIANTS = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]


def splits_ready(seed: int) -> bool:
    return all((SPLITS / s / f"seed_{seed}" / f"{p}.csv").exists()
               for s in ("standard", "zeroshot") for p in ("train", "val", "test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="txgnn", choices=VARIANTS)
    ap.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--pretrain_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sim_loss_weight", type=float, default=0.3)
    ap.add_argument("--skip_leakage_check", action="store_true")
    args = ap.parse_args()

    print("[setup] loading PrimeKG ...")
    edges, entity_idx, graph = load_graph()
    for seed in args.seeds:
        if not splits_ready(seed):
            make_all_splits(edges, seeds=[seed])

    if not args.skip_leakage_check and args.split in ("zeroshot", "both"):
        for seed in args.seeds:
            if audit_split(seed)["status"] == "FAIL":
                sys.exit(f"[ABORT] leakage in zeroshot seed={seed}")
        print("[leakage] zero-shot splits PASS")

    import torch
    from repurpose.util.device import require_cuda
    from repurpose.nets.txgnn_scaled import ScaledTxGNN
    from repurpose.nets.ablation_factory import build_no_attn, build_no_affinity, build_no_both
    from repurpose.train.phase1_pretrain import run_pretrain
    from repurpose.train.txgnn_phase2 import run_txgnn_finetune

    node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
    meta = graph.metadata()
    builders = {
        "txgnn": lambda: ScaledTxGNN(meta, node_counts, hidden_dim=args.hidden_dim,
                                     attention=True, affinity=True),
        "txgnn_no_attn": lambda: build_no_attn(meta, node_counts, hidden_dim=args.hidden_dim),
        "txgnn_no_sim": lambda: build_no_affinity(meta, node_counts, hidden_dim=args.hidden_dim),
        "txgnn_no_both": lambda: build_no_both(meta, node_counts, hidden_dim=args.hidden_dim),
    }

    device = require_cuda(0)
    results_dir = ROOT / "results" / args.variant
    targets = ["standard", "zeroshot"] if args.split == "both" else [args.split]

    for seed in args.seeds:
        for split in targets:
            print(f"\n{'='*60}\n[run] {args.variant} | {split} | seed={seed}\n{'='*60}")
            torch.manual_seed(seed)
            model = builders[args.variant]()
            run_pretrain(model, graph, device, epochs=args.pretrain_epochs, lr=args.lr,
                         results_dir=results_dir, model_name=args.variant, seed=seed)
            result = run_txgnn_finetune(model, graph, entity_idx,
                                        SPLITS / split / f"seed_{seed}", device,
                                        epochs=args.finetune_epochs, lr=args.lr,
                                        results_dir=results_dir, model_name=args.variant,
                                        split_name=split, seed=seed,
                                        sim_loss_weight=args.sim_loss_weight)
            print(json.dumps({k: v for k, v in result.items()
                              if k != "per_disease_results"}, indent=2))
    print(f"\n[done] {results_dir}")


if __name__ == "__main__":
    main()
