"""Phase 1: self-supervised link-prediction pretraining over all edge types.

Shared by the scaled TxGNN, its ablations, and the plain GNN baseline (so the
no-message-passing control still benefits from KG-positioned embeddings).
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


def _corrupt_tails(edge_index, n_dst, device, ratio=1):
    """Negative edges: keep the head, sample a random tail."""
    n = edge_index.size(1)
    tails = torch.randint(0, n_dst, (n * ratio,), device=device)
    heads = edge_index[0].repeat(ratio)
    return torch.stack([heads, tails], dim=0)


def _bce(pos, neg):
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return F.binary_cross_entropy_with_logits(scores, labels)


def run_pretrain(model, graph, device, epochs=50, lr=1e-3, amp=True,
                 results_dir: Path = None, model_name="gnn_baseline", seed=42) -> dict:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=amp)
    edge_index_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}

    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        with autocast("cuda", enabled=amp):
            h = model.embed(edge_index_dict)
            loss = torch.tensor(0.0, device=device)
            seen = 0
            for (src_t, _, dst_t), ei in edge_index_dict.items():
                if ei.size(1) == 0:
                    continue
                pos = (h[src_t][ei[0]] * h[dst_t][ei[1]]).sum(-1)
                neg_ei = _corrupt_tails(ei, graph[dst_t].num_nodes, device)
                neg = (h[src_t][neg_ei[0]] * h[dst_t][neg_ei[1]]).sum(-1)
                loss = loss + _bce(pos, neg)
                seen += 1
            if seen:
                loss = loss / seen

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        elapsed = round(time.time() - start, 1)
        history.append({"epoch": epoch, "loss": float(loss.item()), "elapsed_s": elapsed})
        if epoch == 1 or epoch % 10 == 0:
            print(f"  [phase1] {epoch}/{epochs} loss={loss.item():.4f} t={elapsed:.0f}s")

    summary = {
        "model": model_name,
        "phase": "pretrain",
        "seed": seed,
        "epochs": epochs,
        "final_loss": history[-1]["loss"],
        "total_wall_s": history[-1]["elapsed_s"],
        "epoch_log": history,
    }
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"{model_name}_pretrain_seed{seed}.json"
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  [phase1] log -> {out}")
    return summary
