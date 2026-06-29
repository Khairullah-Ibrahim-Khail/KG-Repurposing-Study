"""Q2 alternative training loops.

run_single_stage
    KG link-prediction loss + therapeutic loss optimized together from epoch 1.

run_contrastive
    Therapeutic loss + InfoNCE disease-similarity loss, no KG pretraining.

Both write the same result schema as the TxGNN runner and tag outputs
``original_ablation``.
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


def _therapeutic_loss(h, train_pairs, n_diseases, drug_type, disease_type, device, ratio):
    loss, n = torch.tensor(0.0, device=device), 0
    for _, (drug_idx, disease_idx) in train_pairs.items():
        drug_idx, disease_idx = drug_idx.to(device), disease_idx.to(device)
        pos = (h[drug_type][drug_idx] * h[disease_type][disease_idx]).sum(-1)
        neg_dis = torch.randint(0, n_diseases, (len(drug_idx) * ratio,), device=device)
        neg = (h[drug_type][drug_idx.repeat(ratio)] * h[disease_type][neg_dis]).sum(-1)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = loss + F.binary_cross_entropy_with_logits(scores, labels)
        n += 1
    return loss / n if n else loss


def _finalize(model, model_name, graph, entity_idx, test_edges, edge_index_dict,
              device, amp, drug_type, disease_type, split_name, seed, wall_s,
              results_dir, extra):
    model.eval()
    with torch.no_grad(), autocast("cuda", enabled=amp):
        h_test = {nt: v.detach() for nt, v in model.embed(edge_index_dict).items()}
    n_diseases = graph[disease_type].num_nodes
    drug_map = entity_idx.get(drug_type, {})
    disease_map = entity_idx.get(disease_type, {})

    test_pairs = pair_tensors(test_edges, drug_map, disease_map)
    flat = flat_metrics(h_test, test_pairs, n_diseases, drug_type, disease_type, device)
    zs = per_disease_eval(model, h_test, test_edges, entity_idx, device, drug_type, disease_type)
    ind = zs["aggregate"].get("indication", {})
    contra = zs["aggregate"].get("contraindication", {})
    cfg = model.config_dict() if hasattr(model, "config_dict") else {}

    result = {
        **envelope(model_name, "original_ablation", split_name, seed, wall_s),
        "model_config": cfg,
        "indication": {"auprc": ind.get("auprc_mean"), "auroc": ind.get("auroc_mean"),
                       "n_test_diseases": ind.get("n_diseases", 0)},
        "contraindication": {"auprc": contra.get("auprc_mean"), "auroc": contra.get("auroc_mean"),
                             "n_test_diseases": contra.get("n_diseases", 0)},
        "indication_flat": {"auprc": flat.get("indication", {}).get("auprc"),
                            "auroc": flat.get("indication", {}).get("auroc"),
                            "n_test_pairs": flat.get("indication", {}).get("n_pairs")},
        "contraindication_flat": {"auprc": flat.get("contraindication", {}).get("auprc"),
                                  "auroc": flat.get("contraindication", {}).get("auroc"),
                                  "n_test_pairs": flat.get("contraindication", {}).get("n_pairs")},
        **extra,
    }
    if results_dir is not None:
        out = results_dir / split_name / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{model_name}.json", "w") as fh:
            json.dump(result, fh, indent=2)
    print(f"  [result] {model_name} | {split_name} | seed={seed} | "
          f"ind={result['indication']['auprc']} wall={wall_s}s")
    return result


def run_single_stage(model, graph, entity_idx, split_dir: Path, device,
                     epochs=100, lr=1e-3, neg_ratio=5, kg_weight=0.5, amp=True,
                     results_dir: Path = None, model_name="single_stage",
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
    test_edges = pd.read_csv(split_dir / "test.csv")

    best_val, best_state, stale, last_epoch = -1.0, None, 0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        opt.zero_grad()
        with autocast("cuda", enabled=amp):
            h = model.embed(edge_index_dict)
            kg_loss, n_kg = torch.tensor(0.0, device=device), 0
            for (src_t, _, dst_t), ei in edge_index_dict.items():
                if ei.size(1) == 0:
                    continue
                pos = (h[src_t][ei[0]] * h[dst_t][ei[1]]).sum(-1)
                neg_tail = torch.randint(0, graph[dst_t].num_nodes, (ei.size(1),), device=device)
                neg = (h[src_t][ei[0]] * h[dst_t][neg_tail]).sum(-1)
                kg_loss = kg_loss + F.binary_cross_entropy_with_logits(
                    torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
                n_kg += 1
            if n_kg:
                kg_loss = kg_loss / n_kg
            task_loss = _therapeutic_loss(h, train_pairs, n_diseases, drug_type, disease_type, device, neg_ratio)
            loss = kg_weight * kg_loss + (1 - kg_weight) * task_loss
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
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _finalize(
        model, model_name, graph, entity_idx, test_edges, edge_index_dict, device, amp,
        drug_type, disease_type, split_name, seed, round(time.time() - start, 1), results_dir,
        extra={"compute": {"kg_loss_weight": kg_weight, "n_epochs_run": last_epoch},
               "notes": f"single_stage|best_val_auprc={best_val:.4f}"},
    )


def run_contrastive(model, graph, entity_idx, split_dir: Path, device,
                    epochs=100, lr=1e-3, neg_ratio=5, contrastive_weight=0.3, amp=True,
                    results_dir: Path = None, model_name="joint_contrastive",
                    split_name="standard", seed=42, patience=10, pairs_per_epoch=128,
                    drug_type="drug", disease_type="disease") -> dict:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=amp)
    edge_index_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}
    n_diseases = graph[disease_type].num_nodes

    drug_map = entity_idx.get(drug_type, {})
    disease_map = entity_idx.get(disease_type, {})
    train_edges = pd.read_csv(split_dir / "train.csv")
    val_pairs = pair_tensors(pd.read_csv(split_dir / "val.csv"), drug_map, disease_map)
    test_edges = pd.read_csv(split_dir / "test.csv")
    train_pairs = pair_tensors(train_edges, drug_map, disease_map)

    # Positive disease pairs: share >= 1 drug among train therapeutic edges.
    disease_drugs = defaultdict(set)
    treat = train_edges[train_edges["relation"].isin(TREATMENT_RELATIONS)]
    for drug_id, disease_id in zip(treat["x_id"], treat["y_id"]):
        if disease_id in disease_map and drug_id in drug_map:
            disease_drugs[disease_map[disease_id]].add(drug_map[drug_id])
    dlist = list(disease_drugs)
    pos_pairs = [(dlist[i], dlist[j]) for i in range(len(dlist)) for j in range(i + 1, len(dlist))
                 if disease_drugs[dlist[i]] & disease_drugs[dlist[j]]]
    print(f"  [contrastive] {len(pos_pairs)} positive disease pairs")
    pos_t = torch.tensor(pos_pairs, dtype=torch.long, device=device) if pos_pairs else None

    best_val, best_state, stale, last_epoch = -1.0, None, 0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        opt.zero_grad()
        with autocast("cuda", enabled=amp):
            h = model.embed(edge_index_dict)
            task_loss = _therapeutic_loss(h, train_pairs, n_diseases, drug_type, disease_type, device, neg_ratio)
            contrast = torch.tensor(0.0, device=device)
            if pos_t is not None and hasattr(model, "infonce_loss"):
                k = min(pairs_per_epoch, pos_t.size(0))
                sample = pos_t[torch.randperm(pos_t.size(0), device=device)[:k]]
                contrast = model.infonce_loss(h[disease_type], sample)
            loss = task_loss + contrastive_weight * contrast
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
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _finalize(
        model, model_name, graph, entity_idx, test_edges, edge_index_dict, device, amp,
        drug_type, disease_type, split_name, seed, round(time.time() - start, 1), results_dir,
        extra={"n_positive_disease_pairs": len(pos_pairs),
               "compute": {"contrastive_weight": contrastive_weight, "n_epochs_run": last_epoch},
               "notes": f"joint_contrastive|best_val_auprc={best_val:.4f}"},
    )
