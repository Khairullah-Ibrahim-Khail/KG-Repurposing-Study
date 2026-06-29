"""Phase 2 for the scaled TxGNN family (full model + Q6 ablations).

Adds the disease-affinity triplet loss on top of the therapeutic task loss, and
evaluates two ways: the random-negative flat protocol (``*_flat`` fields, used
for cross-model comparison) and per-disease zero-shot ranking (used for the
degradation curve and case studies).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from repurpose.eval.zeroshot import per_disease_eval
from repurpose.train._common import (
    TREATMENT_RELATIONS, envelope, flat_metrics, pair_tensors, val_auprc,
)


def _drug_sharing_pairs(train_edges, disease_map, cap=5000) -> torch.Tensor:
    """Disease index pairs that share at least one approved drug (triplet pos)."""
    drug_to_diseases = defaultdict(set)
    treat = train_edges[train_edges["relation"].isin(TREATMENT_RELATIONS)]
    for drug_id, disease_id in zip(treat["x_id"], treat["y_id"]):
        if disease_id in disease_map:
            drug_to_diseases[drug_id].add(disease_map[disease_id])

    pairs = []
    for members in drug_to_diseases.values():
        members = list(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
                if len(pairs) >= cap:
                    break
            if len(pairs) >= cap:
                break
        if len(pairs) >= cap:
            break
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def run_txgnn_finetune(model, graph, entity_idx, split_dir: Path, device,
                       epochs=100, lr=1e-3, neg_ratio=5, amp=True,
                       results_dir: Path = None, model_name="txgnn",
                       split_name="standard", seed=42, patience=10,
                       sim_loss_weight=0.3, drug_type="drug", disease_type="disease") -> dict:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=amp)
    edge_index_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}
    n_diseases = graph[disease_type].num_nodes

    drug_map = entity_idx.get(drug_type, {})
    disease_map = entity_idx.get(disease_type, {})
    train_edges = pd.read_csv(split_dir / "train.csv")
    val_edges = pd.read_csv(split_dir / "val.csv")
    test_edges = pd.read_csv(split_dir / "test.csv")

    train_pairs = pair_tensors(train_edges, drug_map, disease_map)
    val_pairs = pair_tensors(val_edges, drug_map, disease_map)
    pos_pairs = _drug_sharing_pairs(train_edges, disease_map).to(device)

    best_val, best_state, stale = -1.0, None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        with autocast("cuda", enabled=amp):
            h = model.embed(edge_index_dict)
            task_loss, n_tasks = torch.tensor(0.0, device=device), 0
            for _, (drug_idx, disease_idx) in train_pairs.items():
                drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
                pos = (h[drug_type][drug_idx] * h[disease_type][disease_idx]).sum(-1)
                neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * neg_ratio,), device=device)
                neg = (h[drug_type][drug_idx.repeat(neg_ratio)] * h[disease_type][neg_dis]).sum(-1)
                scores = torch.cat([pos, neg])
                labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                task_loss = task_loss + F.binary_cross_entropy_with_logits(scores, labels)
                n_tasks += 1
            if n_tasks:
                task_loss = task_loss / n_tasks

            sim_loss = torch.tensor(0.0, device=device)
            if model.affinity is not None and pos_pairs.size(0) > 0 and sim_loss_weight > 0:
                k = min(pos_pairs.size(0), 500)
                sample = pos_pairs[torch.randperm(pos_pairs.size(0), device=device)[:k]]
                neg_pairs = torch.stack([
                    torch.randint(0, n_diseases, (k,), device=device),
                    torch.randint(0, n_diseases, (k,), device=device),
                ], dim=1)
                sim_loss = model.affinity.triplet_loss(h[disease_type], sample, neg_pairs)

            loss = task_loss + sim_loss_weight * sim_loss

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
            if stale >= patience and epoch > 20:
                print(f"  [txgnn] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=amp):
        h_test = {nt: v.detach() for nt, v in model.embed(edge_index_dict).items()}

    test_pairs = pair_tensors(test_edges, drug_map, disease_map)
    flat = flat_metrics(h_test, test_pairs, n_diseases, drug_type, disease_type, device)
    zs = per_disease_eval(model, h_test, test_edges, entity_idx, device, drug_type, disease_type)
    wall_s = round(time.time() - start, 1)

    cfg = model.config_dict() if hasattr(model, "config_dict") else {}
    ind = zs["aggregate"].get("indication", {})
    contra = zs["aggregate"].get("contraindication", {})

    result = {
        **envelope(model_name, cfg.get("reproduction_type", "scaled_reproduction"),
                   split_name, seed, wall_s),
        "model_config": cfg,
        "indication": {"auprc": ind.get("auprc_mean"), "auprc_std": ind.get("auprc_std"),
                       "auroc": ind.get("auroc_mean"), "n_test_diseases": ind.get("n_diseases", 0)},
        "contraindication": {"auprc": contra.get("auprc_mean"), "auprc_std": contra.get("auprc_std"),
                             "auroc": contra.get("auroc_mean"),
                             "n_test_diseases": contra.get("n_diseases", 0)},
        "per_disease_results": zs["per_disease"],
        "indication_flat": {"auprc": flat.get("indication", {}).get("auprc"),
                            "auroc": flat.get("indication", {}).get("auroc"),
                            "n_test_pairs": flat.get("indication", {}).get("n_pairs")},
        "contraindication_flat": {"auprc": flat.get("contraindication", {}).get("auprc"),
                                  "auroc": flat.get("contraindication", {}).get("auroc"),
                                  "n_test_pairs": flat.get("contraindication", {}).get("n_pairs")},
        "notes": f"best_val_auprc={best_val:.4f}",
    }

    if results_dir is not None:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  [txgnn] -> {out / f'{model_name}.json'}")

    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind={result['indication']['auprc']} contra={result['contraindication']['auprc']} "
          f"wall={wall_s}s")
    return result
