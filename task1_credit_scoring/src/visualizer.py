"""
Visualization module – generates publication-quality plots.

All figures are saved to the ``outputs/`` directory.
"""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


# ── global style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#ef4444"]
MODEL_COLORS = {}


def generate_all_plots(
    results: Dict[str, Any],
    feature_names: List[str],
    output_dir: str = "outputs",
) -> None:
    """Create and save all four visualizations.

    Parameters
    ----------
    results : dict
        Output of ``evaluate_models()``.
    feature_names : list[str]
        Feature names for the importance chart.
    output_dir : str
        Directory to write PNG files into.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = [k for k in results if k != "best_model"]
    global MODEL_COLORS
    MODEL_COLORS = {name: PALETTE[i] for i, name in enumerate(model_names)}

    _plot_roc_curves(results, model_names, output_dir)
    _plot_confusion_matrices(results, model_names, output_dir)
    _plot_feature_importance(results, model_names, feature_names, output_dir)
    _plot_model_comparison(results, model_names, output_dir)

    print(f"\n📊  All plots saved to {output_dir}/")


# ── individual plots ────────────────────────────────────────────────────────


def _plot_roc_curves(results: dict, model_names: list, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in model_names:
        r = results[name]
        ax.plot(
            r["fpr"],
            r["tpr"],
            label=f'{name} (AUC={r["roc_auc"]:.3f})',
            color=MODEL_COLORS[name],
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Credit Default Prediction", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close(fig)


def _plot_confusion_matrices(results: dict, model_names: list, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, name in zip(axes, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Good", "Default"],
            yticklabels=["Good", "Default"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(
    results: dict, model_names: list, feature_names: list, output_dir: str
) -> None:
    # use the best model's importances
    best = results.get("best_model", model_names[-1])
    importances = results[best].get("feature_importances", {})
    if not importances:
        # fallback to Gradient Boosting if best has none
        for name in reversed(model_names):
            importances = results[name].get("feature_importances", {})
            if importances:
                best = name
                break

    names = list(importances.keys())
    values = list(importances.values())
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        np.array(names)[order],
        np.array(values)[order],
        color=sns.color_palette("viridis", len(names)),
        edgecolor="white",
    )
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Feature Importance – {best}", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200)
    plt.close(fig)


def _plot_model_comparison(results: dict, model_names: list, output_dir: str) -> None:
    metrics = ["accuracy", "f1_score", "roc_auc", "cv_auc_mean"]
    metric_labels = ["Accuracy", "F1 Score", "ROC-AUC", "CV AUC (mean)"]

    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=name,
            color=MODEL_COLORS[name],
            edgecolor="white",
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – All Metrics", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print("Visualizer module – import and call generate_all_plots().")
