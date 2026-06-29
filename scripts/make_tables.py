"""Q5 + Q6 — assemble comparison tables straight from the result JSONs.

Outputs (never edit by hand — regenerate):
    results/metrics/comparison_table.csv
    results/metrics/q6_ablation_table.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1]
SPLITS = ["standard", "zeroshot"]

# model -> function(split, seed) -> result path
MODEL_PATHS = {
    "gnn_no_kg": lambda sp, s: RESULTS / "gnn" / sp / f"seed_{s}" / "gnn_no_kg.json",
    "gnn_kg": lambda sp, s: RESULTS / "gnn" / sp / f"seed_{s}" / "gnn_baseline.json",
    "txgnn_two_phase": lambda sp, s: RESULTS / "txgnn" / sp / f"seed_{s}" / "txgnn.json",
    "single_stage": lambda sp, s: RESULTS / "alt_single_stage" / sp / f"seed_{s}" / "single_stage.json",
    "joint_contrastive": lambda sp, s: RESULTS / "alt_joint_contrastive" / sp / f"seed_{s}" / "joint_contrastive.json",
    "txgnn_attn_on": lambda sp, s: RESULTS / "ablations" / "txgnn" / sp / f"seed_{s}" / "txgnn.json",
    "txgnn_attn_off": lambda sp, s: RESULTS / "ablations" / "txgnn_no_attn" / sp / f"seed_{s}" / "txgnn_no_attn.json",
    "transformer_kg": lambda sp, s: RESULTS / "transformer_kg" / sp / f"seed_{s}" / "transformer_kg.json",
    "transformer_nokg": lambda sp, s: RESULTS / "transformer_nokg" / sp / f"seed_{s}" / "transformer_nokg.json",
}


def load(path: Path, fallback: Path = None):
    if not path.exists() and fallback is not None and fallback.exists():
        path = fallback
    return json.loads(path.read_text()) if path.exists() else None


def metric(result: dict, field: str, sub: str):
    """Prefer the flat random-negative value; fall back to the plain block."""
    flat = result.get(f"{field}_flat")
    if flat and flat.get(sub) is not None:
        return flat[sub]
    return result.get(field, {}).get(sub)


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def fmt(value):
    return round(value, 4) if value is not None else "[NOT YET RUN]"


def comparison_table() -> pd.DataFrame:
    rows = []
    for model, path_of in MODEL_PATHS.items():
        for split in SPLITS:
            fb = (lambda s, sp=split: RESULTS / "txgnn" / sp / f"seed_{s}" / "txgnn.json") \
                if model == "txgnn_attn_on" else None
            loaded = [load(path_of(split, s), fb(s) if fb else None) for s in SEEDS]
            have = [r for r in loaded if r is not None]
            base = {
                "model": model,
                "split": split,
                "n_seeds_run": len(have),
                "reproduction_type": have[0].get("reproduction_type", "?") if have else "?",
            }
            if not have:
                for col in ("auprc_ind", "auroc_ind", "auprc_contra", "auroc_contra", "wall_s"):
                    base[col] = "[NOT YET RUN]"
                    if col != "wall_s":
                        base[col + "_std"] = ""
                rows.append(base)
                continue
            ia, ia_s = mean_std([metric(r, "indication", "auprc") for r in have])
            iu, iu_s = mean_std([metric(r, "indication", "auroc") for r in have])
            ca, ca_s = mean_std([metric(r, "contraindication", "auprc") for r in have])
            cu, cu_s = mean_std([metric(r, "contraindication", "auroc") for r in have])
            clocks = [r.get("wall_clock_seconds") for r in have if r.get("wall_clock_seconds")]
            base.update({
                "auprc_ind": fmt(ia), "auprc_ind_std": fmt(ia_s),
                "auroc_ind": fmt(iu), "auroc_ind_std": fmt(iu_s),
                "auprc_contra": fmt(ca), "auprc_contra_std": fmt(ca_s),
                "auroc_contra": fmt(cu), "auroc_contra_std": fmt(cu_s),
                "wall_s": round(float(np.mean(clocks)), 1) if clocks else "[NOT YET RUN]",
            })
            rows.append(base)
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "comparison_table.csv", index=False)
    print(f"[table] {METRICS / 'comparison_table.csv'}")
    print(df.to_string(index=False))
    return df


def q6_table() -> pd.DataFrame:
    rows = []
    for variant in ("txgnn_attn_on", "txgnn_attn_off"):
        path_of = MODEL_PATHS[variant]
        for split in SPLITS:
            fb = (lambda s, sp=split: RESULTS / "txgnn" / sp / f"seed_{s}" / "txgnn.json") \
                if variant == "txgnn_attn_on" else None
            have = [r for r in (load(path_of(split, s), fb(s) if fb else None) for s in SEEDS)
                    if r is not None]
            vals = [metric(r, "indication", "auprc") for r in have]
            vals = [v for v in vals if v is not None]
            rows.append({
                "variant": variant, "split": split,
                "auprc_ind_mean": round(float(np.mean(vals)), 4) if vals else None,
                "auprc_ind_std": round(float(np.std(vals)), 4) if vals else None,
                "n_seeds": len(vals),
            })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "q6_ablation_table.csv", index=False)
    print(f"[q6] {METRICS / 'q6_ablation_table.csv'}")
    return df


def main() -> None:
    print("=== Q5 comparison ===")
    comparison_table()
    print("\n=== Q6 ablation ===")
    q6_table()


if __name__ == "__main__":
    sys.exit(main())
