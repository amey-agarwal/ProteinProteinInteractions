"""Generate ROC and PR curve plots from saved per-mode .npz curve files."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CURVES_DIR = Path("data/processed/curves")
FIGURES_DIR = Path("figures")

MODES = ["sequence", "network", "sequence_network"]
COLORS = {
    "sequence": "tab:blue",
    "network": "tab:orange",
    "sequence_network": "tab:green",
    "evidence": "tab:red",
    "combined": "tab:purple",
}


def plot_roc(ax, modes):
    for mode in modes:
        path = CURVES_DIR / f"{mode}.npz"
        if not path.exists():
            print(f"Skipping {mode} (no curves file)")
            continue
        data = np.load(path)
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f"{mode} (AUC = {float(data['roc_auc']):.3f})",
            color=COLORS.get(mode, "black"),
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — PPI Prediction")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def plot_pr(ax, modes):
    prevalence = None
    for mode in modes:
        path = CURVES_DIR / f"{mode}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        ax.plot(
            data["recall"],
            data["precision"],
            label=f"{mode} (AP = {float(data['pr_auc']):.3f})",
            color=COLORS.get(mode, "black"),
            linewidth=2,
        )
        if prevalence is None:
            prevalence = float(data["y_test"].mean())

    if prevalence is not None:
        ax.axhline(prevalence, ls="--", color="k", alpha=0.3, label=f"Prevalence ({prevalence:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — PPI Prediction")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    available = [m for m in MODES if (CURVES_DIR / f"{m}.npz").exists()]
    if not available:
        raise FileNotFoundError(f"No curve files found under {CURVES_DIR}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    plot_roc(ax, available)
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'roc_curves.png'}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    plot_pr(ax, available)
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'pr_curves.png'}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_roc(axes[0], available)
    plot_pr(axes[1], available)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_combined.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'roc_pr_combined.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
