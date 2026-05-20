"""
Visualizer Module
=================
Generates training curve plots and confusion matrix heatmaps.
Saves all figures to the outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary": "#7C3AED",
    "secondary": "#06B6D4",
    "accent": "#F59E0B",
    "bg": "#1E1B2E",
    "text": "#E2E8F0",
}


def plot_training_curves(
    history: Dict[str, list],
    model_name: str,
    output_dir: str,
) -> str:
    """
    Plot training and validation accuracy/loss curves.

    Parameters
    ----------
    history : dict
        Training history with keys: loss, val_loss, accuracy, val_accuracy.
    model_name : str
        Model name for the title.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    filepath : str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    epochs = range(1, len(history.get("loss", [])) + 1)

    # --- Accuracy Plot ---
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    if "accuracy" in history:
        ax.plot(epochs, history["accuracy"], color=COLORS["secondary"],
                linewidth=2, label="Train Accuracy", marker="o", markersize=3)
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], color=COLORS["accent"],
                linewidth=2, label="Val Accuracy", marker="s", markersize=3)
    ax.set_title(f"{model_name} - Accuracy", color=COLORS["text"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", color=COLORS["text"])
    ax.set_ylabel("Accuracy", color=COLORS["text"])
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["primary"],
              labelcolor=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.grid(alpha=0.3)

    # --- Loss Plot ---
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    if "loss" in history:
        ax.plot(epochs, history["loss"], color=COLORS["primary"],
                linewidth=2, label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], color=COLORS["accent"],
                linewidth=2, label="Val Loss", marker="s", markersize=3)
    ax.set_title(f"{model_name} - Loss", color=COLORS["text"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", color=COLORS["text"])
    ax.set_ylabel("Loss", color=COLORS["text"])
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["primary"],
              labelcolor=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"training_curves_{safe_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Training curves saved: {filepath}")
    return filepath


def plot_confusion_matrix(
    cm: np.ndarray,
    emotion_names: List[str],
    model_name: str,
    output_dir: str,
) -> str:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes x n_classes).
    emotion_names : list of str
        Class labels.
    model_name : str
        Model name for the title.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    filepath : str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # Normalize confusion matrix for color mapping
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Create heatmap
    cmap = sns.color_palette("blend:#1E1B2E,#7C3AED,#06B6D4", as_cmap=True)
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show raw counts
        fmt="d",
        cmap=cmap,
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        ax=ax,
        linewidths=1,
        linecolor=COLORS["bg"],
        cbar_kws={"label": "Proportion"},
        annot_kws={"size": 12, "color": COLORS["text"]},
    )

    ax.set_title(f"{model_name} - Confusion Matrix",
                 color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("True", color=COLORS["text"], fontsize=12)
    ax.tick_params(colors=COLORS["text"])

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=COLORS["text"])
    cbar.set_label("Proportion", color=COLORS["text"])

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Confusion matrix saved: {filepath}")
    return filepath


def plot_model_comparison(
    comparison: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Plot a bar chart comparing model performance metrics.

    Parameters
    ----------
    comparison : dict
        Model comparison data from evaluator.compare_models().
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    filepath : str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    models_data = comparison["models"]
    model_names = [m["name"] for m in models_data]
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 (Macro)", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    x = np.arange(len(model_names))
    width = 0.18
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#EC4899"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors_list)):
        values = [m[metric] for m in models_data]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    color=COLORS["text"], fontsize=9)

    ax.set_xlabel("Model", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("Score", color=COLORS["text"], fontsize=12)
    ax.set_title("Model Comparison", color=COLORS["text"], fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["primary"],
              labelcolor=COLORS["text"])
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Model comparison saved: {filepath}")
    return filepath


if __name__ == "__main__":
    # Quick test
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

    # Test training curves
    history = {
        "accuracy": [0.3, 0.5, 0.65, 0.72, 0.78],
        "val_accuracy": [0.28, 0.45, 0.58, 0.65, 0.70],
        "loss": [1.8, 1.2, 0.9, 0.7, 0.55],
        "val_loss": [1.9, 1.3, 1.0, 0.8, 0.65],
    }
    plot_training_curves(history, "TestModel", output_dir)

    # Test confusion matrix
    cm = np.array([
        [8, 1, 0, 0, 1, 0],
        [0, 9, 0, 0, 1, 0],
        [1, 0, 7, 1, 0, 1],
        [0, 0, 1, 8, 0, 1],
        [0, 1, 0, 0, 9, 0],
        [0, 0, 1, 1, 0, 8],
    ])
    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
    plot_confusion_matrix(cm, emotions, "TestModel", output_dir)

    print("Visualizer test complete.")
