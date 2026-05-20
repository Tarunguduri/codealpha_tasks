"""
Evaluator Module
================
Computes classification metrics: accuracy, per-class report, confusion matrix.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Optional


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained Keras model.
        x_test: Test images (N, 28, 28, 1), normalized.
        y_test: True test labels (N,).
        class_names: Optional list of class label names.

    Returns:
        Dictionary containing:
            - model_name: Name of the model
            - test_accuracy: Overall accuracy on test set
            - test_loss: Loss on test set
            - predictions: Predicted class labels
            - prediction_probs: Prediction probabilities for each class
            - confusion_matrix: Confusion matrix as nested list
            - classification_report: Per-class precision/recall/f1 as dict
            - correct_indices: Indices of correctly classified samples
            - incorrect_indices: Indices of misclassified samples
    """
    model_name = model.name
    print(f"\n[Evaluator] Evaluating {model_name} on test set...")

    # Get predictions
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    prediction_probs = model.predict(x_test, verbose=0)
    predictions = np.argmax(prediction_probs, axis=1)

    # Compute metrics
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    target_names = class_names if class_names else [str(i) for i in range(len(np.unique(y_test)))]
    report = classification_report(
        y_test, predictions, target_names=target_names, output_dict=True
    )

    # Find correct and incorrect predictions
    correct_mask = predictions == y_test
    correct_indices = np.where(correct_mask)[0].tolist()
    incorrect_indices = np.where(~correct_mask)[0].tolist()

    results = {
        "model_name": model_name,
        "test_accuracy": float(acc),
        "test_loss": float(test_loss),
        "predictions": predictions.tolist(),
        "prediction_probs": prediction_probs.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "correct_indices": correct_indices[:100],  # Cap for JSON size
        "incorrect_indices": incorrect_indices[:100],
        "n_correct": len(correct_indices),
        "n_incorrect": len(incorrect_indices),
        "total_samples": len(y_test),
    }

    # Print summary
    print(f"  Test Accuracy: {acc:.4f} ({results['n_correct']}/{results['total_samples']})")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Misclassified: {results['n_incorrect']} samples")

    return results


def compare_models(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple model evaluation results.

    Args:
        results_list: List of evaluation result dictionaries.

    Returns:
        Comparison dictionary with model names and metrics.
    """
    comparison = {
        "models": [],
        "best_model": None,
        "best_accuracy": 0.0,
    }

    print(f"\n{'='*60}")
    print(f"  Model Comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Loss':>10} {'Errors':>10}")
    print(f"  {'-'*50}")

    for result in results_list:
        model_info = {
            "name": result["model_name"],
            "accuracy": result["test_accuracy"],
            "loss": result["test_loss"],
            "n_errors": result["n_incorrect"],
        }
        comparison["models"].append(model_info)

        if result["test_accuracy"] > comparison["best_accuracy"]:
            comparison["best_accuracy"] = result["test_accuracy"]
            comparison["best_model"] = result["model_name"]

        print(f"  {model_info['name']:<20} {model_info['accuracy']:>10.4f} "
              f"{model_info['loss']:>10.4f} {model_info['n_errors']:>10}")

    print(f"\n  🏆 Best Model: {comparison['best_model']} "
          f"({comparison['best_accuracy']:.4f} accuracy)")
    print(f"{'='*60}\n")

    return comparison


if __name__ == "__main__":
    print("Evaluator module loaded successfully.")
    print("Available functions: evaluate_model(), compare_models()")
