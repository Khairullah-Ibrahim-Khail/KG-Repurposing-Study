"""Q6 — build results/ablations/matrix.json from the ablation result files.

The Q6 verdict compares txgnn vs txgnn_no_attn on zero-shot indication AUPRC
against a +/-0.02 threshold fixed before running.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ABLATIONS = ROOT / "results" / "ablations"
OUT = ABLATIONS / "matrix.json"

VARIANTS = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]
SPLITS = ["standard", "zeroshot"]
SEEDS = [42, 0, 1]
TASKS = ["indication", "contraindication"]
THRESHOLD = 0.02


def load(variant: str, split: str, seed: int):
    path = ABLATIONS / variant / split / f"seed_{seed}" / f"{variant}.json"
    if not path.exists() and variant == "txgnn":
        path = ROOT / "results" / "txgnn" / split / f"seed_{seed}" / "txgnn.json"
    return json.loads(path.read_text()) if path.exists() else None


def _auprc(r, task):
    flat = r.get(f"{task}_flat")
    if flat and flat.get("auprc") is not None:
        return flat["auprc"]
    return r.get(task, {}).get("auprc")


def _auroc(r, task):
    flat = r.get(f"{task}_flat")
    if flat and flat.get("auroc") is not None:
        return flat["auroc"]
    return r.get(task, {}).get("auroc")


def aggregate(variant: str, split: str) -> dict:
    have = [r for r in (load(variant, split, s) for s in SEEDS) if r is not None]
    row = {"variant": variant, "split": split, "n_seeds": len(have)}
    if not have:
        for task in TASKS:
            row[f"{task}_auprc_mean"] = "[NOT YET RUN]"
            row[f"{task}_auprc_std"] = "[NOT YET RUN]"
            row[f"{task}_auroc_mean"] = "[NOT YET RUN]"
        row["wall_clock_s_mean"] = "[NOT YET RUN]"
        return row
    for task in TASKS:
        aps = [v for v in (_auprc(r, task) for r in have) if v is not None]
        rcs = [v for v in (_auroc(r, task) for r in have) if v is not None]
        row[f"{task}_auprc_mean"] = round(float(np.mean(aps)), 5) if aps else None
        row[f"{task}_auprc_std"] = round(float(np.std(aps)), 5) if len(aps) > 1 else None
        row[f"{task}_auroc_mean"] = round(float(np.mean(rcs)), 5) if rcs else None
    clocks = [r.get("wall_clock_seconds") for r in have if r.get("wall_clock_seconds")]
    row["wall_clock_s_mean"] = round(float(np.mean(clocks)), 1) if clocks else None
    return row


def verdict(rows: list) -> dict:
    full = next((r for r in rows if r["variant"] == "txgnn" and r["split"] == "zeroshot"), None)
    off = next((r for r in rows if r["variant"] == "txgnn_no_attn" and r["split"] == "zeroshot"), None)
    if not full or not off:
        return {"status": "NOT_YET_RUN", "detail": "missing variant"}
    a, b = full.get("indication_auprc_mean"), off.get("indication_auprc_mean")
    if a in (None, "[NOT YET RUN]") or b in (None, "[NOT YET RUN]"):
        return {"status": "NOT_YET_RUN", "detail": "AUPRC unavailable"}
    delta = round(float(a) - float(b), 5)
    if delta >= THRESHOLD:
        label = "BENEFICIAL — attention improves zero-shot AUPRC"
    elif delta >= -THRESHOLD:
        label = "OPTIONAL — removing attention has minimal effect"
    else:
        label = "DETRIMENTAL — removing attention IMPROVES zero-shot AUPRC"
    return {
        "status": "DECIDED",
        "txgnn_attn_on_zeroshot_ind_auprc": a,
        "txgnn_attn_off_zeroshot_ind_auprc": b,
        "delta_attn_on_minus_off": delta,
        "threshold": THRESHOLD,
        "attention_optional": abs(delta) < THRESHOLD,
        "conclusion": f"Attention is {label}. Delta = {delta:.5f}.",
    }


def main() -> None:
    rows = [aggregate(v, s) for v in VARIANTS for s in SPLITS]
    matrix = {
        "generated_from": "scripts/ablation_matrix.py",
        "source_dir": str(ABLATIONS),
        "seeds": SEEDS,
        "q6_decision": verdict(rows),
        "matrix": rows,
    }
    ABLATIONS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matrix, indent=2))
    print(f"[matrix] {OUT}")
    print(json.dumps(matrix["q6_decision"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
