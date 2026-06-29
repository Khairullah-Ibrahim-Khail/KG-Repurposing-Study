"""Q6 — train all four ablation variants.

    python scripts/run_ablations.py --variant all --split both

Writes results/ablations/<variant>/<split>/seed_<n>/<variant>.json.
Then run scripts/ablation_matrix.py to build the decision matrix.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from repurpose.graph.kg_ingest import load_graph
from repurpose.graph.split_builder import make_all_splits

SPLITS = ROOT / "data" / "splits"
VARIANTS = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]


def splits_ready(seed: int) -> bool:
    return all((SPLITS / s / f"seed_{seed}" / f"{p}.csv").exists()
               for s in ("standard", "zeroshot") for p in ("train", "val", "test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=VARIANTS + ["all"])
    ap.add_argument("--split", default="both", choices=["standard", "zeroshot", "both"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--hidden_dim", type=int, default=64)
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
    from repurpose.nets.txgnn_scaled import ScaledTxGNN
    from repurpose.nets.ablation_factory import build_no_attn, build_no_affinity, build_no_both
    from repurpose.train.phase1_pretrain import run_pretrain
    from repurpose.train.txgnn_phase2 import run_txgnn_finetune

    node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
    meta = graph.metadata()

    def make(variant):
        if variant == "txgnn":
            return ScaledTxGNN(meta, node_counts, hidden_dim=args.hidden_dim,
                               attention=True, affinity=True)
        if variant == "txgnn_no_attn":
            return build_no_attn(meta, node_counts, hidden_dim=args.hidden_dim)
        if variant == "txgnn_no_sim":
            return build_no_affinity(meta, node_counts, hidden_dim=args.hidden_dim)
        return build_no_both(meta, node_counts, hidden_dim=args.hidden_dim)

    device = require_cuda(0)
    variants = VARIANTS if args.variant == "all" else [args.variant]
    targets = ["standard", "zeroshot"] if args.split == "both" else [args.split]

    for variant in variants:
        results_dir = ROOT / "results" / "ablations" / variant
        for seed in args.seeds:
            for split in targets:
                print(f"\n{'='*60}\n[ablation] {variant} | {split} | seed={seed}")
                torch.manual_seed(seed)
                model = make(variant)
                run_pretrain(model, graph, device, epochs=args.pretrain_epochs, lr=args.lr,
                             results_dir=results_dir, model_name=variant, seed=seed)
                run_txgnn_finetune(model, graph, entity_idx,
                                   SPLITS / split / f"seed_{seed}", device,
                                   epochs=args.finetune_epochs, lr=args.lr,
                                   results_dir=results_dir, model_name=variant,
                                   split_name=split, seed=seed)
    print("\n[done] run scripts/ablation_matrix.py next.")


if __name__ == "__main__":
    main()
