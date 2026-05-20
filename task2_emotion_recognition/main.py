"""
Emotion Recognition from Speech - Main Pipeline
=================================================
Orchestrates the complete ML pipeline:
1. Load data (RAVDESS or synthetic fallback)
2. Extract features (librosa or synthetic fallback)
3. Split data (70/15/15) with proper StandardScaler fitting
4. Train models (CNN, Bi-LSTM with TF, or MLP with sklearn)
5. Evaluate and compare models
6. Generate visualizations and save results

CRITICAL FIX: StandardScaler is fit ONLY on training data, then used
to transform validation and test sets. This prevents data leakage.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_ravdess,
    generate_synthetic_features,
    SELECTED_EMOTIONS,
    EMOTION_TO_LABEL,
    N_FEATURES,
)
from src.feature_extractor import (
    extract_features_batch,
    is_librosa_available,
)
from src.models import (
    build_cnn,
    build_lstm,
    build_mlp,
    is_tf_available,
    reshape_for_cnn,
    reshape_for_lstm,
)
from src.trainer import train_keras_model, train_sklearn_model
from src.evaluator import evaluate_model, compare_models
from src.visualizer import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
)


def main():
    """Run the complete emotion recognition pipeline."""
    print("=" * 70)
    print("  🎙️  Emotion Recognition from Speech")
    print("  " + "=" * 66)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TensorFlow available: {is_tf_available()}")
    print(f"  librosa available:    {is_librosa_available()}")
    print("=" * 70 + "\n")

    # --- Configuration ---
    RAVDESS_DIR = os.path.join(PROJECT_ROOT, "data", "RAVDESS")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RANDOM_SEED = 42
    N_PER_CLASS = 200
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15  # of total → 0.15 / 0.85 ≈ 0.176 of train+val
    EPOCHS = 100
    BATCH_SIZE = 32

    np.random.seed(RANDOM_SEED)

    # =========================================================
    # STEP 1: Load Data
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 1: Loading Data")
    print("=" * 50)

    use_synthetic = False
    X, y = None, None

    # Try loading RAVDESS dataset
    file_paths, labels, emotion_names = load_ravdess(RAVDESS_DIR)

    if file_paths is not None and is_librosa_available():
        # Extract features from real audio
        print("\n[Pipeline] Extracting features from RAVDESS audio files...")
        X, y = extract_features_batch(file_paths, labels)

    if X is None:
        # Fallback to synthetic data
        print("\n[Pipeline] Using synthetic data for demo mode.")
        use_synthetic = True
        X, y, emotion_names = generate_synthetic_features(
            n_per_class=N_PER_CLASS,
            n_features=N_FEATURES,
            random_seed=RANDOM_SEED,
        )

    print(f"\n[Pipeline] Dataset: X={X.shape}, y={y.shape}")
    print(f"[Pipeline] Mode: {'Synthetic (Demo)' if use_synthetic else 'RAVDESS (Real Audio)'}")

    # =========================================================
    # STEP 2: Train/Val/Test Split (70/15/15)
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 2: Splitting Data (70/15/15)")
    print("=" * 50)

    # First split: 85% train+val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Second split: from 85%, take ~17.6% for val → 15% of total
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.15 / 0.85 ≈ 0.176
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # =========================================================
    # STEP 3: Feature Scaling (FIT ON TRAIN ONLY - prevents leakage!)
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 3: Feature Scaling (StandardScaler)")
    print("  ⚠️  CRITICAL: Scaler fit ONLY on training data!")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # FIT + TRANSFORM on train
    X_val_scaled = scaler.transform(X_val)            # TRANSFORM ONLY on val
    X_test_scaled = scaler.transform(X_test)          # TRANSFORM ONLY on test

    print(f"  Train mean range: [{X_train_scaled.mean(axis=0).min():.4f}, "
          f"{X_train_scaled.mean(axis=0).max():.4f}]")
    print(f"  Train std range:  [{X_train_scaled.std(axis=0).min():.4f}, "
          f"{X_train_scaled.std(axis=0).max():.4f}]")

    # Save the scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"  Saved scaler to outputs/scaler.pkl")

    # =========================================================
    # STEP 4: Train Models
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 4: Training Models")
    print("=" * 50)

    n_classes = len(SELECTED_EMOTIONS)
    n_features = X_train_scaled.shape[1]
    all_results = []
    all_histories = {}

    if is_tf_available():
        # --- CNN Model ---
        print("\n--- Training CNN ---")
        X_train_cnn = reshape_for_cnn(X_train_scaled)
        X_val_cnn = reshape_for_cnn(X_val_scaled)
        X_test_cnn = reshape_for_cnn(X_test_scaled)

        cnn_model = build_cnn((n_features, 1), n_classes)
        cnn_result = train_keras_model(
            cnn_model, X_train_cnn, y_train, X_val_cnn, y_val,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
        )
        all_histories["CNN"] = cnn_result["history"]

        # Evaluate CNN
        cnn_eval = evaluate_model(
            cnn_result["model"], X_test_cnn, y_test,
            emotion_names, model_name="CNN", is_keras=True,
        )
        cnn_eval["training_time"] = cnn_result["training_time"]
        cnn_eval["epochs_trained"] = cnn_result["epochs_trained"]
        all_results.append(cnn_eval)

        # Visualize CNN
        plot_training_curves(cnn_result["history"], "CNN", OUTPUT_DIR)
        plot_confusion_matrix(
            np.array(cnn_eval["confusion_matrix"]),
            emotion_names, "CNN", OUTPUT_DIR,
        )

        # --- Bi-LSTM Model ---
        print("\n--- Training Bi-LSTM ---")
        X_train_lstm = reshape_for_lstm(X_train_scaled)
        X_val_lstm = reshape_for_lstm(X_val_scaled)
        X_test_lstm = reshape_for_lstm(X_test_scaled)

        lstm_model = build_lstm((1, n_features), n_classes)
        lstm_result = train_keras_model(
            lstm_model, X_train_lstm, y_train, X_val_lstm, y_val,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
        )
        all_histories["Bi-LSTM"] = lstm_result["history"]

        # Evaluate Bi-LSTM
        lstm_eval = evaluate_model(
            lstm_result["model"], X_test_lstm, y_test,
            emotion_names, model_name="Bi-LSTM", is_keras=True,
        )
        lstm_eval["training_time"] = lstm_result["training_time"]
        lstm_eval["epochs_trained"] = lstm_result["epochs_trained"]
        all_results.append(lstm_eval)

        # Save Bi-LSTM model
        lstm_result["model"].save(os.path.join(OUTPUT_DIR, "Bi-LSTM.keras"))

        # Visualize Bi-LSTM
        plot_training_curves(lstm_result["history"], "Bi-LSTM", OUTPUT_DIR)
        plot_confusion_matrix(
            np.array(lstm_eval["confusion_matrix"]),
            emotion_names, "Bi-LSTM", OUTPUT_DIR,
        )

    # --- MLP (always available via sklearn) ---
    print("\n--- Training MLP (sklearn) ---")
    mlp_model = build_mlp(n_classes)
    mlp_result = train_sklearn_model(
        mlp_model, X_train_scaled, y_train, X_val_scaled, y_val,
    )
    all_histories["MLP"] = mlp_result["history"]

    # Evaluate MLP
    mlp_eval = evaluate_model(
        mlp_result["model"], X_test_scaled, y_test,
        emotion_names, model_name="MLP", is_keras=False,
    )
    mlp_eval["training_time"] = mlp_result["training_time"]
    mlp_eval["epochs_trained"] = mlp_result["epochs_trained"]
    all_results.append(mlp_eval)

    # Visualize MLP
    plot_training_curves(mlp_result["history"], "MLP", OUTPUT_DIR)
    plot_confusion_matrix(
        np.array(mlp_eval["confusion_matrix"]),
        emotion_names, "MLP", OUTPUT_DIR,
    )

    # =========================================================
    # STEP 5: Compare Models
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 5: Model Comparison")
    print("=" * 50)

    comparison = compare_models(all_results)
    plot_model_comparison(comparison, OUTPUT_DIR)

    # =========================================================
    # STEP 6: Save Results to JSON
    # =========================================================
    print("\n" + "=" * 50)
    print("  STEP 6: Saving Results")
    print("=" * 50)

    # Build comprehensive results dict
    results_json = {
        "metadata": {
            "project": "Emotion Recognition from Speech",
            "timestamp": datetime.now().isoformat(),
            "data_mode": "synthetic" if use_synthetic else "ravdess",
            "n_samples": int(len(X)),
            "n_features": int(n_features),
            "n_classes": n_classes,
            "emotions": emotion_names,
            "split": {
                "train": int(X_train.shape[0]),
                "val": int(X_val.shape[0]),
                "test": int(X_test.shape[0]),
            },
            "scaler": "StandardScaler (fit on train only)",
            "tensorflow_available": is_tf_available(),
            "librosa_available": is_librosa_available(),
        },
        "models": {},
        "comparison": comparison,
        "training_histories": all_histories,
    }

    # Add per-model results
    for res in all_results:
        name = res["model_name"]
        results_json["models"][name] = {
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "f1_weighted": res["f1_weighted"],
            "precision_macro": res["precision_macro"],
            "recall_macro": res["recall_macro"],
            "per_emotion_accuracy": res["per_emotion_accuracy"],
            "confusion_matrix": res["confusion_matrix"],
            "classification_report": res["classification_report"],
            "training_time": res.get("training_time", 0),
            "epochs_trained": res.get("epochs_trained", 0),
            "n_test_samples": res["n_test_samples"],
        }

    # Save JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Results saved to: {results_path}")

    # =========================================================
    # Final Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("  🎉 Pipeline Complete!")
    print("=" * 70)
    print(f"  Best Model:    {comparison['best_model']}")
    print(f"  Best Accuracy: {comparison['best_accuracy']:.4f} "
          f"({comparison['best_accuracy']*100:.1f}%)")
    print(f"  Output Dir:    {OUTPUT_DIR}")
    print(f"  Dashboard:     Open dashboard/index.html in a browser")
    print("=" * 70)

    return results_json


if __name__ == "__main__":
    results = main()
