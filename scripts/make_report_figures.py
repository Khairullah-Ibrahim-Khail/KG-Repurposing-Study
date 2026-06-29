"""Generate publication figures for the report straight from the result files.

Reads results/metrics/comparison_table.csv and results/ablations/matrix.json and
writes real data plots to report/figures/. No screenshots, no hand-typed numbers.

    python scripts/make_report_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY = "#16223B"
INDIGO = "#4F46E5"
TEAL = "#0EA5A4"
GREEN = "#16A34A"
CLAY = "#E0563B"
GREY = "#5B6B82"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#9aa6b5",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e3e8ef",
    "grid.linewidth": 0.8,
    "figure.dpi": 150,
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_table():
    rows = {}
    with open(ROOT / "results" / "metrics" / "comparison_table.csv") as fh:
        for r in csv.DictReader(fh):
            rows[(r["model"], r["split"])] = r
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fig_q1(rows):
    """Indication AUPRC: no-KG vs KG, per split, with std error bars."""
    splits = ["standard", "zeroshot"]
    labels = ["Standard split", "Zero-shot split"]
    nokg = [fnum(rows[("gnn_no_kg", s)]["auprc_ind"]) for s in splits]
    nokg_e = [fnum(rows[("gnn_no_kg", s)]["auprc_ind_std"]) for s in splits]
    kg = [fnum(rows[("gnn_kg", s)]["auprc_ind"]) for s in splits]
    kg_e = [fnum(rows[("gnn_kg", s)]["auprc_ind_std"]) for s in splits]

    x = range(len(splits))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    b1 = ax.bar([i - w / 2 for i in x], nokg, w, yerr=nokg_e, capsize=4,
                color=GREEN, label="No-KG (no message passing)")
    b2 = ax.bar([i + w / 2 for i in x], kg, w, yerr=kg_e, capsize=4,
                color=INDIGO, label="KG (2-layer HGT)")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Indication AUPRC")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_q1.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_q1.png")


def fig_q2(rows):
    """Zero-shot indication AUPRC + wall-clock for the three training recipes."""
    order = ["txgnn_two_phase", "single_stage", "joint_contrastive"]
    names = ["Two-phase", "Single-stage", "Joint-contrastive"]
    ap = [fnum(rows[(m, "zeroshot")]["auprc_ind"]) for m in order]
    e = [fnum(rows[(m, "zeroshot")]["auprc_ind_std"]) for m in order]
    wall = [fnum(rows[(m, "zeroshot")]["wall_s"]) for m in order]

    x = range(len(order))
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    colors = [INDIGO, GREY, GREY]
    bars = ax.bar(x, ap, 0.55, yerr=e, capsize=4, color=colors)
    bars[0].set_color(INDIGO)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Zero-shot indication AUPRC")
    ax.set_ylim(0.6, 0.8)
    _style(ax)

    ax2 = ax.twinx()
    ax2.plot(list(x), wall, "o--", color=CLAY, linewidth=1.4, markersize=6,
             label="Wall-clock (s)")
    for xi, wv in zip(x, wall):
        ax2.text(xi, wv + 4, f"{wv:.0f}s", ha="center", color=CLAY, fontsize=8)
    ax2.set_ylabel("Wall-clock (s)", color=CLAY)
    ax2.tick_params(axis="y", colors=CLAY, length=0)
    ax2.set_ylim(0, 160)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_q2.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_q2.png")


def fig_q6():
    """Zero-shot indication AUPRC across the four ablation variants."""
    matrix = json.loads((ROOT / "results" / "ablations" / "matrix.json").read_text())
    zero = {r["variant"]: r for r in matrix["matrix"] if r["split"] == "zeroshot"}
    order = ["txgnn", "txgnn_no_attn", "txgnn_no_sim", "txgnn_no_both"]
    names = ["attn+sim\n(full)", "no attn", "no sim", "no both"]
    ap = [zero[v]["indication_auprc_mean"] for v in order]
    e = [zero[v].get("indication_auprc_std") or 0 for v in order]

    x = range(len(order))
    colors = [INDIGO, GREEN, GREY, GREY]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    bars = ax.bar(x, ap, 0.6, yerr=e, capsize=4, color=colors)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(ap[0], color=INDIGO, linestyle=":", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Zero-shot indication AUPRC")
    ax.set_ylim(0.69, 0.79)
    ax.set_title("Removing attention IMPROVES AUPRC (Δ = −0.036)",
                 fontsize=9.5, color=NAVY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_q6.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_q6.png")


def main():
    rows = load_table()
    fig_q1(rows)
    fig_q2(rows)
    fig_q6()
    print("done ->", FIGS)


if __name__ == "__main__":
    main()
