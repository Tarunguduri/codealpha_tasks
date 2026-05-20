"""
Handwritten Character Recognition — Main Pipeline
==================================================
Orchestrates the full ML pipeline: data loading → preprocessing → training →
evaluation → visualization → dashboard data export.

Usage:
    python main.py

Configuration:
    USE_EMNIST: Set to True to use EMNIST Letters (26 classes), False for MNIST (10 digits).
    EPOCHS: Number of training epochs.
"""

import os
import sys
import json
import time


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import numpy as np

# Project root for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_mnist, load_emnist_letters, get_dataset_summary
from src.preprocessor import preprocess_pipeline
from src.models import build_simple_cnn, build_deep_cnn, get_model_summary_dict
from src.trainer import train_model
from src.evaluator import evaluate_model, compare_models
from src.visualizer import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_model_comparison,
)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
USE_EMNIST = False          # False = MNIST digits, True = EMNIST letters
EPOCHS = 15                 # Training epochs
BATCH_SIZE = 128            # Training batch size
VAL_FRACTION = 0.1          # 10% stratified validation split (FIXED)
USE_AUGMENTATION = True     # ImageDataGenerator augmentation
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def main():
    """Run the complete handwritten character recognition pipeline."""
    start_time = time.time()

    print("\n" + "═" * 60)
    print("  🖊️  Handwritten Character Recognition Pipeline")
    print("═" * 60)

    # ── Step 1: Create output directory ──────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[Pipeline] Output directory: {OUTPUT_DIR}")

    # ── Step 2: Load data ────────────────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 1: Loading Dataset")
    print("─" * 40)

    if USE_EMNIST:
        x_train, y_train, x_test, y_test, metadata = load_emnist_letters()
    else:
        x_train, y_train, x_test, y_test, metadata = load_mnist()

    print(get_dataset_summary(metadata))

    # ── Step 3: Preprocess ───────────────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 2: Preprocessing")
    print("─" * 40)

    x_train_proc, x_val, y_train_proc, y_val, x_test_proc = preprocess_pipeline(
        x_train, y_train, x_test, val_fraction=VAL_FRACTION
    )

    print(f"\n  Final shapes:")
    print(f"    x_train: {x_train_proc.shape}")
    print(f"    x_val:   {x_val.shape}")
    print(f"    x_test:  {x_test_proc.shape}")

    # ── Step 4: Build models ─────────────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 3: Building Models")
    print("─" * 40)

    n_classes = metadata["n_classes"]
    input_shape = (28, 28, 1)

    simple_cnn = build_simple_cnn(input_shape=input_shape, n_classes=n_classes)
    deep_cnn = build_deep_cnn(input_shape=input_shape, n_classes=n_classes)

    # ── Step 5: Train models ─────────────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 4: Training Models")
    print("─" * 40)

    simple_train_result = train_model(
        simple_cnn, x_train_proc, y_train_proc, x_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, use_augmentation=USE_AUGMENTATION,
    )

    deep_train_result = train_model(
        deep_cnn, x_train_proc, y_train_proc, x_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, use_augmentation=USE_AUGMENTATION,
    )

    # ── Step 6: Save model weights ───────────────────────────
    print("\n[Pipeline] Saving model weights...")
    simple_weights_path = os.path.join(OUTPUT_DIR, "simple_cnn.weights.h5")
    deep_weights_path = os.path.join(OUTPUT_DIR, "deep_cnn.weights.h5")
    simple_cnn.save_weights(simple_weights_path)
    deep_cnn.save_weights(deep_weights_path)
    print(f"  ✓ SimpleCNN weights: {simple_weights_path}")
    print(f"  ✓ DeepCNN weights: {deep_weights_path}")

    # ── Step 7: Evaluate models ──────────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 5: Evaluating Models")
    print("─" * 40)

    simple_eval = evaluate_model(simple_cnn, x_test_proc, y_test, metadata["class_names"])
    deep_eval = evaluate_model(deep_cnn, x_test_proc, y_test, metadata["class_names"])

    comparison = compare_models([simple_eval, deep_eval])

    # ── Step 8: Generate visualizations ──────────────────────
    print("\n" + "─" * 40)
    print("  STEP 6: Generating Visualizations")
    print("─" * 40)

    # Training curves
    plot_training_curves(simple_train_result["history"], "SimpleCNN", OUTPUT_DIR)
    plot_training_curves(deep_train_result["history"], "DeepCNN", OUTPUT_DIR)

    # Confusion matrices
    plot_confusion_matrix(
        np.array(simple_eval["confusion_matrix"]),
        metadata["class_names"], "SimpleCNN", OUTPUT_DIR,
    )
    plot_confusion_matrix(
        np.array(deep_eval["confusion_matrix"]),
        metadata["class_names"], "DeepCNN", OUTPUT_DIR,
    )

    # Sample predictions
    predictions_simple = np.array(simple_eval["predictions"])
    predictions_deep = np.array(deep_eval["predictions"])

    plot_sample_predictions(
        x_test_proc, y_test, predictions_simple,
        metadata["class_names"], "SimpleCNN", OUTPUT_DIR,
    )
    plot_sample_predictions(
        x_test_proc, y_test, predictions_deep,
        metadata["class_names"], "DeepCNN", OUTPUT_DIR,
    )

    # Model comparison
    plot_model_comparison(comparison, OUTPUT_DIR)

    # ── Step 9: Save results to JSON ─────────────────────────
    print("\n" + "─" * 40)
    print("  STEP 7: Saving Results")
    print("─" * 40)

    # Prepare sample predictions for dashboard (first 50 of each type)
    sample_predictions = []
    for i in range(min(50, len(y_test))):
        sample_predictions.append({
            "index": i,
            "true_label": metadata["class_names"][int(y_test[i])],
            "simple_pred": metadata["class_names"][int(predictions_simple[i])],
            "deep_pred": metadata["class_names"][int(predictions_deep[i])],
            "simple_correct": bool(predictions_simple[i] == y_test[i]),
            "deep_correct": bool(predictions_deep[i] == y_test[i]),
        })

    elapsed = time.time() - start_time

    results = {
        "project": "Handwritten Character Recognition",
        "dataset": metadata["dataset_name"],
        "n_classes": metadata["n_classes"],
        "class_names": metadata["class_names"],
        "train_samples": int(metadata["train_samples"]),
        "test_samples": int(metadata["test_samples"]),
        "validation_fraction": VAL_FRACTION,
        "epochs": EPOCHS,
        "use_augmentation": USE_AUGMENTATION,
        "elapsed_seconds": round(elapsed, 1),
        "models": {
            "SimpleCNN": {
                "training": simple_train_result,
                "evaluation": {
                    "test_accuracy": simple_eval["test_accuracy"],
                    "test_loss": simple_eval["test_loss"],
                    "n_correct": simple_eval["n_correct"],
                    "n_incorrect": simple_eval["n_incorrect"],
                    "confusion_matrix": simple_eval["confusion_matrix"],
                    "classification_report": simple_eval["classification_report"],
                },
                "architecture": get_model_summary_dict(simple_cnn),
            },
            "DeepCNN": {
                "training": deep_train_result,
                "evaluation": {
                    "test_accuracy": deep_eval["test_accuracy"],
                    "test_loss": deep_eval["test_loss"],
                    "n_correct": deep_eval["n_correct"],
                    "n_incorrect": deep_eval["n_incorrect"],
                    "confusion_matrix": deep_eval["confusion_matrix"],
                    "classification_report": deep_eval["classification_report"],
                },
                "architecture": get_model_summary_dict(deep_cnn),
            },
        },
        "comparison": comparison,
        "sample_predictions": sample_predictions,
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  ✓ Results saved: {results_path}")

    # ── Final Summary ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Dataset:         {metadata['dataset_name']}")
    print(f"  Classes:         {metadata['n_classes']}")
    print(f"  SimpleCNN Acc:   {simple_eval['test_accuracy']:.4f}")
    print(f"  DeepCNN Acc:     {deep_eval['test_accuracy']:.4f}")
    print(f"  Best Model:      {comparison['best_model']}")
    print(f"  Total Time:      {elapsed:.1f}s")
    print(f"  Output Dir:      {OUTPUT_DIR}")
    print(f"  Dashboard:       dashboard/index.html")
    print("═" * 60 + "\n")

    return results


if __name__ == "__main__":
    main()
