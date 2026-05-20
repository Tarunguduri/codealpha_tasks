"""
Credit Scoring ML Pipeline – Main Entry Point.

Orchestrates data generation, preprocessing, model training, evaluation,
visualization, and JSON export.
"""

import json
import os
import sys
import time
import joblib
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_credit_data
from src.preprocessor import preprocess_data
from src.models import build_models, train_models
from src.evaluator import evaluate_models
from src.visualizer import generate_all_plots

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    """Run the full credit scoring pipeline."""
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Data Generation ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 1 · Generating synthetic credit data")
    print("═" * 60)
    df = generate_credit_data(n_samples=5000, random_state=42)
    print(f"  Samples : {len(df)}")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Default rate: {df['default'].mean():.2%}")

    # save raw data
    df.to_csv(OUTPUT_DIR / "credit_data.csv", index=False)
    print(f"  Saved raw data → outputs/credit_data.csv")

    # ── 2. Preprocessing ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2 · Preprocessing & Feature Engineering")
    print("═" * 60)
    data = preprocess_data(df, target_col="default", test_size=0.2, random_state=42)

    # ── 3. Model Building ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 3 · Building & Training Models")
    print("═" * 60)
    # We pass RAW (unscaled) data because each Pipeline has its own scaler,
    # preventing data-leakage from the preprocessor's global scaler.
    pipelines = build_models()
    pipelines = train_models(pipelines, data["X_train_raw"], data["y_train"])

    # ── 4. Evaluation ───────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 4  Evaluating Models")
    print(" ?" * 60)
    results = evaluate_models(
        pipelines,
        X_train=data["X_train_raw"],
        X_test=data["X_test_raw"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        feature_names=data["feature_names"],
    )
    
    # Save the best pipeline (scaler + model)
    best_pipeline = pipelines[results["best_model"]]
    joblib.dump(best_pipeline, OUTPUT_DIR / "best_model.pkl")
    print(f"  Saved best model ({results['best_model']}) ' {OUTPUT_DIR / 'best_model.pkl'}")

    # ── 5. Visualization ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 5 · Generating Visualizations")
    print("═" * 60)
    generate_all_plots(results, data["feature_names"], output_dir=str(OUTPUT_DIR))

    # ── 6. Save Results JSON ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 6 · Exporting Results")
    print("═" * 60)
    _save_results_json(results, data["feature_names"])

    elapsed = time.time() - start
    print("\n" + "═" * 60)
    print(f"  ✅  Pipeline complete in {elapsed:.1f}s")
    print(f"  📁  All outputs saved to: {OUTPUT_DIR}")
    print("═" * 60 + "\n")


def _save_results_json(results: dict, feature_names: list) -> None:
    """Write a dashboard-friendly JSON file."""
    output = {
        "best_model": results["best_model"],
        "feature_names": feature_names,
        "models": {},
    }
    for name, data in results.items():
        if name == "best_model":
            continue
        output["models"][name] = {
            "accuracy": data["accuracy"],
            "f1_score": data["f1_score"],
            "roc_auc": data["roc_auc"],
            "cv_auc_mean": data["cv_auc_mean"],
            "cv_auc_std": data["cv_auc_std"],
            "confusion_matrix": data["confusion_matrix"],
            "feature_importances": data["feature_importances"],
            "fpr": data["fpr"],
            "tpr": data["tpr"],
        }
    path = OUTPUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
