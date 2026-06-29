"""Phase 2 for the plain GNN baseline: fine-tune on therapeutic edges with a
binary classification objective (no affinity head, no metric loss).

Writes ``results/gnn/<split>/seed_<n>/<model_name>.json``.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from repurpose.train._common import (
    envelope, flat_metrics, pair_tensors, val_auprc,
)


def run_finetune(model, graph, entity_idx, split_dir: Path, device,
                 epochs=100, lr=1e-3, neg_ratio=5, amp=True,
                 results_dir: Path = None, model_name="gnn_baseline",
                 split_name="standard", seed=42, patience=10,
                 drug_type="drug", disease_type="disease") -> dict:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=amp)
    edge_index_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}
    n_diseases = graph[disease_type].num_nodes

    drug_map = entity_idx.get(drug_type, {})
    disease_map = entity_idx.get(disease_type, {})
    train_pairs = pair_tensors(pd.read_csv(split_dir / "train.csv"), drug_map, disease_map)
    val_pairs = pair_tensors(pd.read_csv(split_dir / "val.csv"), drug_map, disease_map)
    test_pairs = pair_tensors(pd.read_csv(split_dir / "test.csv"), drug_map, disease_map)

    best_val, best_state, stale = -1.0, None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        with autocast("cuda", enabled=amp):
            h = model.embed(edge_index_dict)
            loss, n_tasks = torch.tensor(0.0, device=device), 0
            for _, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos = (h[drug_type][drug_idx] * h[disease_type][disease_idx]).sum(-1)
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
                neg_drug = drug_idx.repeat(neg_ratio)
                neg = (h[drug_type][neg_drug] * h[disease_type][neg_dis]).sum(-1)
                scores = torch.cat([pos, neg])
                labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                loss = loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1
            if n_tasks:
                loss = loss / n_tasks
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad(), autocast("cuda", enabled=amp):
                h_eval = model.embed(edge_index_dict)
            score = val_auprc(h_eval, val_pairs, n_diseases, drug_type, disease_type, device)
            if score > best_val:
                best_val, stale = score, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 5
            if stale >= patience:
                print(f"  [phase2] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=amp):
        h_test = {nt: v.detach() for nt, v in model.embed(edge_index_dict).items()}
    metrics = flat_metrics(h_test, test_pairs, n_diseases, drug_type, disease_type, device)
    wall_s = round(time.time() - start, 1)

    ind = metrics.get("indication", {})
    contra = metrics.get("contraindication", {})
    result = {
        **envelope(model_name, "scaled_reproduction", split_name, seed, wall_s),
        "indication": {"auprc": ind.get("auprc"), "auroc": ind.get("auroc"),
                       "n_test_pairs": ind.get("n_pairs", 0)},
        "contraindication": {"auprc": contra.get("auprc"), "auroc": contra.get("auroc"),
                             "n_test_pairs": contra.get("n_pairs", 0)},
        "notes": f"best_val_auprc={best_val:.4f}",
    }

    if results_dir is not None:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  [phase2] -> {out / f'{model_name}.json'}")

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind={result['indication']['auprc']} contra={result['contraindication']['auprc']} "
          f"wall={wall_s}s")
    return result
