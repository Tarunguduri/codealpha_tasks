"""
Visualizer Module
=================
Generates publication-quality plots: training curves, confusion matrices,
sample predictions, and model comparison charts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#0a0a1a",
    "axes.facecolor": "#0a0a1a",
    "axes.edgecolor": "#333366",
    "axes.labelcolor": "#e0e0ff",
    "text.color": "#e0e0ff",
    "xtick.color": "#9999cc",
    "ytick.color": "#9999cc",
    "grid.color": "#1a1a3a",
    "figure.dpi": 150,
    "font.size": 10,
})


def plot_training_curves(
    history: Dict[str, list],
    model_name: str,
    output_dir: str = "outputs",
) -> str:
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Training history dict with keys: accuracy, val_accuracy, loss, val_loss.
        model_name: Name of the model for the title.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["accuracy"]) + 1)

    # Accuracy plot
    ax1.plot(epochs, history["accuracy"], "o-", color="#00d4ff", label="Train Accuracy",
             linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_accuracy"], "s-", color="#ff6b9d", label="Val Accuracy",
             linewidth=2, markersize=4)
    ax1.set_title(f"{model_name} — Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor="#1a1a3a", edgecolor="#333366")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.9, 1.01] if max(history["accuracy"]) > 0.9 else [0, 1.05])

    # Loss plot
    ax2.plot(epochs, history["loss"], "o-", color="#00d4ff", label="Train Loss",
             linewidth=2, markersize=4)
    ax2.plot(epochs, history["val_loss"], "s-", color="#ff6b9d", label="Val Loss",
             linewidth=2, markersize=4)
    ax2.set_title(f"{model_name} — Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#1a1a3a", edgecolor="#333366")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"training_curves_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Training curves saved: {filepath}")
    return filepath


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: str = "outputs",
) -> str:
    """
    Plot a confusion matrix heatmap. Adapts figure size for 10 vs 26 classes.

    Args:
        cm: Confusion matrix array (n_classes × n_classes).
        class_names: List of class label names.
        model_name: Model name for title.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    # Adapt figure size based on number of classes
    if n_classes <= 10:
        figsize = (10, 8)
        fontsize = 11
        annot = True
        fmt = "d"
    else:
        figsize = (16, 14)
        fontsize = 7
        annot = True
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap (dark blue → neon cyan)
    cmap = sns.color_palette("mako", as_cmap=True)

    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#1a1a3a",
        annot_kws={"size": fontsize},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Confusion matrix saved: {filepath}")
    return filepath


def plot_sample_predictions(
    x_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: str = "outputs",
    n_rows: int = 4,
    n_cols: int = 5,
) -> str:
    """
    Plot a grid of sample predictions with color-coded labels.
    Green = correct, Red = incorrect.

    Args:
        x_test: Test images.
        y_test: True labels.
        predictions: Predicted labels.
        class_names: Class names for display.
        model_name: Model name for title.
        output_dir: Directory to save.
        n_rows: Grid rows (default: 4).
        n_cols: Grid columns (default: 5).

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = n_rows * n_cols

    # Select a mix of correct and incorrect predictions
    correct_mask = predictions == y_test
    incorrect_indices = np.where(~correct_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    # Take some incorrect and fill rest with correct
    n_incorrect = min(len(incorrect_indices), n_samples // 3)
    n_correct = n_samples - n_incorrect

    rng = np.random.RandomState(42)
    selected = np.concatenate([
        rng.choice(incorrect_indices, size=n_incorrect, replace=False) if n_incorrect > 0 else np.array([], dtype=int),
        rng.choice(correct_indices, size=n_correct, replace=False),
    ]).astype(int)
    rng.shuffle(selected)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    fig.suptitle(f"{model_name} — Sample Predictions", fontsize=16, fontweight="bold", y=1.02)

    for idx, ax in enumerate(axes.flat):
        if idx < len(selected):
            sample_idx = selected[idx]
            image = x_test[sample_idx].squeeze()
            true_label = y_test[sample_idx]
            pred_label = predictions[sample_idx]
            is_correct = true_label == pred_label

            ax.imshow(image, cmap="gray")
            color = "#00ff88" if is_correct else "#ff4466"
            symbol = "✓" if is_correct else "✗"
            ax.set_title(
                f"{symbol} P:{class_names[pred_label]} T:{class_names[true_label]}",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"sample_predictions_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Sample predictions saved: {filepath}")
    return filepath


def plot_model_comparison(
    comparison: Dict[str, Any],
    output_dir: str = "outputs",
) -> str:
    """
    Plot a bar chart comparing model accuracies.

    Args:
        comparison: Comparison dict from evaluator.compare_models().
        output_dir: Directory to save.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = [m["name"] for m in comparison["models"]]
    accuracies = [m["accuracy"] for m in comparison["models"]]
    losses = [m["loss"] for m in comparison["models"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy bars
    colors = ["#00d4ff", "#ff6b9d", "#9d4eff", "#00ff88"]
    bars1 = ax1.bar(names, accuracies, color=colors[:len(names)], edgecolor="#333366",
                    linewidth=1.5, alpha=0.85)
    ax1.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([min(accuracies) - 0.01, 1.005])

    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{acc:.4f}", ha="center", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax1.grid(True, alpha=0.2, axis="y")

    # Loss bars
    bars2 = ax2.bar(names, losses, color=colors[:len(names)], edgecolor="#333366",
                    linewidth=1.5, alpha=0.85)
    ax2.set_title("Model Loss Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Loss")

    for bar, loss in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{loss:.4f}", ha="center", fontsize=11, fontweight="bold", color="#e0e0ff")

    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualizer] Model comparison chart saved: {filepath}")
    return filepath


if __name__ == "__main__":
    print("Visualizer module loaded successfully.")
    print("Available functions:")
    print("  - plot_training_curves()")
    print("  - plot_confusion_matrix()")
    print("  - plot_sample_predictions()")
    print("  - plot_model_comparison()")
