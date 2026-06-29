"""Q3 — degradation curve: per-disease AUPRC vs. training treatment-edge count.

Outputs:
    results/metrics/degradation_curve_data.json
    results/figures/degradation_curve.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
SEED = 42

# label -> (results_subdir, result_filename, split used for edge counts)
SOURCES = {
    "gnn_standard": (ROOT / "results" / "gnn", "gnn_baseline", "standard"),
    "gnn_zeroshot": (ROOT / "results" / "gnn", "gnn_baseline", "zeroshot"),
    "txgnn_zeroshot": (ROOT / "results" / "txgnn", "txgnn", "zeroshot"),
}


def per_disease(results_dir: Path, fname: str, split: str) -> list:
    path = results_dir / split / f"seed_{SEED}" / f"{fname}.json"
    if not path.exists():
        print(f"[warn] missing {path}")
        return []
    return json.loads(path.read_text()).get("per_disease_results", [])


def edge_counts(split: str) -> dict:
    path = ROOT / "data" / "splits" / split / f"seed_{SEED}" / "train.csv"
    if not path.exists():
        return {}
    train = pd.read_csv(path)
    treat = train[train["relation"].isin(["indication", "contraindication"])]
    return {str(k): int(v) for k, v in treat.groupby("y_id").size().to_dict().items()}


def build_records() -> list:
    records = []
    for label, (results_dir, fname, split) in SOURCES.items():
        counts = edge_counts(split)
        for row in per_disease(results_dir, fname, split):
            records.append({
                "model": label,
                "disease_id": str(row["disease_id"]),
                "relation": row["relation"],
                "n_train_edges": counts.get(str(row["disease_id"]), 0),
                "auprc": row["auprc"],
            })
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "degradation_curve_data.json").write_text(json.dumps(records, indent=2))
    print(f"[data] {METRICS / 'degradation_curve_data.json'}")
    return records


def plot(records: list) -> None:
    if not records:
        print("[skip] nothing to plot")
        return
    df = pd.DataFrame(records)
    FIGURES.mkdir(parents=True, exist_ok=True)
    bins = [0, 1, 5, 10, 25, 50, 200]
    labels = ["0", "1-4", "5-9", "10-24", "25-49", "50+"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rel in zip(axes, ["indication", "contraindication"]):
        sub = df[df["relation"] == rel].copy()
        if sub.empty:
            ax.set_title(f"{rel} (no data)")
            continue
        sub["bin"] = pd.cut(sub["n_train_edges"], bins=bins, labels=labels, right=False)
        for model in sub["model"].unique():
            grouped = (sub[sub["model"] == model]
                       .groupby("bin", observed=True)["auprc"]
                       .agg(["mean", "std", "count"]).reset_index())
            ax.plot(grouped["bin"].astype(str), grouped["mean"], marker="o",
                    label=f"{model} (n={int(grouped['count'].sum())})")
            ax.fill_between(grouped["bin"].astype(str),
                            grouped["mean"] - grouped["std"].fillna(0),
                            grouped["mean"] + grouped["std"].fillna(0), alpha=0.15)
        ax.set_title(f"{rel.capitalize()} — AUPRC vs training edges (seed={SEED})")
        ax.set_xlabel("training treatment edges per disease")
        ax.set_ylabel("AUPRC")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "degradation_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] {FIGURES / 'degradation_curve.png'}")


def main() -> None:
    plot(build_records())


if __name__ == "__main__":
    sys.exit(main())
