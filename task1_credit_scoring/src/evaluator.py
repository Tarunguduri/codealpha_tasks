"""
Model evaluation utilities.

Computes accuracy, F1, ROC-AUC, confusion matrices, classification reports,
and 5-fold stratified cross-validation scores.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from typing import Any, Dict, List


def evaluate_models(
    pipelines: Dict[str, Pipeline],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Evaluate every fitted pipeline on the test set and via cross-validation.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Fitted model pipelines.
    X_train, X_test : np.ndarray
        Raw (unscaled) feature arrays — the pipelines scale internally.
    y_train, y_test : np.ndarray
        Label arrays.
    feature_names : list[str]
        Feature names for importance extraction.
    cv_folds : int
        Number of stratified CV folds.

    Returns
    -------
    dict
        Nested results per model, plus overall best model info.
    """
    results: Dict[str, Any] = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, pipe in pipelines.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # cross-validation on training data
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # feature importances (where available)
        importances = _extract_importances(pipe, feature_names)

        # classification report
        report = classification_report(y_test, y_pred, target_names=["Good", "Default"])
        print(report)
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances,
        }

    # determine best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    results["best_model"] = best_name
    print(f"\n🏆  Best model: {best_name} (AUC = {results[best_name]['roc_auc']:.4f})")

    return results


def _extract_importances(
    pipe: Pipeline, feature_names: List[str]
) -> Dict[str, float]:
    """Extract feature importances from the classifier inside a pipeline."""
    clf = pipe.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return {}
    imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
    return {name: round(float(v), 4) for name, v in zip(feature_names, imp_norm)}


if __name__ == "__main__":
    print("Evaluator module – import and call evaluate_models().")
