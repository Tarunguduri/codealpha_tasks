"""
Evaluator Module
================
Computes evaluation metrics for emotion recognition models:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .models import is_tf_available


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    emotion_names: List[str],
    model_name: str = "Model",
    is_keras: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : object
        Trained model (Keras or sklearn).
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels (integers).
    emotion_names : list of str
        Emotion class names for the report.
    model_name : str
        Name of the model for display.
    is_keras : bool
        Whether the model is a Keras model.

    Returns
    -------
    results : dict
        Evaluation results including accuracy, per-class metrics,
        confusion matrix, and classification report string.
    """
    # Get predictions
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)

    # Overall metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    report_str = classification_report(
        y_test, y_pred,
        target_names=emotion_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test, y_pred,
        target_names=emotion_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Per-emotion accuracy
    per_emotion_accuracy = {}
    for i, emotion in enumerate(emotion_names):
        mask = y_test == i
        if mask.sum() > 0:
            per_emotion_accuracy[emotion] = float(np.mean(y_pred[mask] == y_test[mask]))
        else:
            per_emotion_accuracy[emotion] = 0.0

    # Print results
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 (macro):       {f1_macro:.4f}")
    print(f"  F1 (weighted):    {f1_weighted:.4f}")
    print(f"  Precision (macro):{precision_macro:.4f}")
    print(f"  Recall (macro):   {recall_macro:.4f}")
    print(f"\n{report_str}")
    print(f"{'='*60}\n")

    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "per_emotion_accuracy": per_emotion_accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_str": report_str,
        "n_test_samples": int(len(y_test)),
    }

    return results


def compare_models(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple model evaluation results.

    Parameters
    ----------
    results_list : list of dict
        List of evaluation result dicts from evaluate_model().

    Returns
    -------
    comparison : dict
        Summary comparison of all models.
    """
    comparison = {
        "models": [],
        "best_model": None,
        "best_accuracy": 0.0,
    }

    print("\n" + "=" * 60)
    print("  Model Comparison Summary")
    print("=" * 60)
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'F1 (macro)':>12s} {'Precision':>12s} {'Recall':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for res in results_list:
        name = res["model_name"]
        acc = res["accuracy"]
        f1 = res["f1_macro"]
        prec = res["precision_macro"]
        rec = res["recall_macro"]

        print(f"  {name:<20s} {acc:>10.4f} {f1:>12.4f} {prec:>12.4f} {rec:>10.4f}")

        comparison["models"].append({
            "name": name,
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
        })

        if acc > comparison["best_accuracy"]:
            comparison["best_accuracy"] = acc
            comparison["best_model"] = name

    print(f"\n  🏆 Best Model: {comparison['best_model']} "
          f"(Accuracy: {comparison['best_accuracy']:.4f})")
    print("=" * 60 + "\n")

    return comparison


if __name__ == "__main__":
    # Quick test with dummy data
    from sklearn.dummy import DummyClassifier

    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
    X_test = np.random.randn(60, 10)
    y_test = np.repeat(np.arange(6), 10)

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_test, y_test)

    results = evaluate_model(dummy, X_test, y_test, emotions, "DummyModel")
    print(f"Test accuracy: {results['accuracy']:.4f}")
