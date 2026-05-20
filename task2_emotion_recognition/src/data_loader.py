"""
Data Loader Module
==================
Handles RAVDESS dataset loading and synthetic feature generation for demo mode.

RAVDESS filename format: 03-01-[EMOTION]-01-01-01-[ACTOR].wav
Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
Selected emotions: neutral, happy, sad, angry, fearful, disgust
"""

import os
import glob
import numpy as np
from typing import Tuple, Dict, List, Optional

# Emotion mapping from RAVDESS codes to labels
EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Emotions we want to classify
SELECTED_EMOTIONS: List[str] = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

# Label encoding for selected emotions
EMOTION_TO_LABEL: Dict[str, int] = {e: i for i, e in enumerate(SELECTED_EMOTIONS)}
LABEL_TO_EMOTION: Dict[int, str] = {i: e for e, i in EMOTION_TO_LABEL.items()}

# Total features: 40 MFCCs * 2 (mean+std) + 12 Chroma + 128 Mel + 1 ZCR + 1 RMS = 222
N_FEATURES: int = 222


def load_ravdess(data_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Load audio file paths and labels from the RAVDESS dataset directory.

    Parameters
    ----------
    data_dir : str
        Path to the RAVDESS dataset root directory.
        Expected structure: data_dir/Actor_XX/03-01-XX-XX-XX-XX-XX.wav

    Returns
    -------
    file_paths : np.ndarray or None
        Array of audio file paths, or None if no files found.
    labels : np.ndarray or None
        Integer labels for each file, or None if no files found.
    emotion_names : list of str
        List of selected emotion names.
    """
    if not os.path.exists(data_dir):
        print(f"[DataLoader] RAVDESS directory not found: {data_dir}")
        return None, None, SELECTED_EMOTIONS

    # Search for .wav files recursively
    pattern = os.path.join(data_dir, "**", "*.wav")
    wav_files = glob.glob(pattern, recursive=True)

    if not wav_files:
        print(f"[DataLoader] No .wav files found in {data_dir}")
        return None, None, SELECTED_EMOTIONS

    file_paths = []
    labels = []

    for fpath in sorted(wav_files):
        fname = os.path.basename(fpath)
        parts = fname.replace(".wav", "").split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion_name = EMOTION_MAP.get(emotion_code)

        if emotion_name and emotion_name in SELECTED_EMOTIONS:
            file_paths.append(fpath)
            labels.append(EMOTION_TO_LABEL[emotion_name])

    if not file_paths:
        print("[DataLoader] No matching emotion files found in RAVDESS dataset.")
        return None, None, SELECTED_EMOTIONS

    print(f"[DataLoader] Loaded {len(file_paths)} files from RAVDESS dataset.")
    file_paths_arr = np.array(file_paths)
    labels_arr = np.array(labels, dtype=np.int64)

    # Print class distribution
    for emotion in SELECTED_EMOTIONS:
        idx = EMOTION_TO_LABEL[emotion]
        count = np.sum(labels_arr == idx)
        print(f"  {emotion:>10s}: {count} samples")

    return file_paths_arr, labels_arr, SELECTED_EMOTIONS


def generate_synthetic_features(
    n_per_class: int = 200,
    n_features: int = N_FEATURES,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic feature vectors for demo/testing when RAVDESS data
    or librosa is not available.

    Each emotion class gets a distinct statistical signature to make
    classification non-trivial but learnable.

    Parameters
    ----------
    n_per_class : int
        Number of samples per emotion class.
    n_features : int
        Number of features per sample (default 222 to match real extraction).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Synthetic feature matrix.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    emotion_names : list of str
        List of selected emotion names.
    """
    rng = np.random.RandomState(random_seed)
    n_classes = len(SELECTED_EMOTIONS)

    # Create distinct cluster centers for each emotion
    # Each emotion gets a unique mean shift pattern across feature groups
    emotion_profiles = {
        "neutral":  {"mean_shift": 0.0, "energy": 0.3, "pitch_var": 0.2, "tempo": 0.5},
        "happy":    {"mean_shift": 1.5, "energy": 0.9, "pitch_var": 0.8, "tempo": 0.9},
        "sad":      {"mean_shift": -1.0, "energy": 0.2, "pitch_var": 0.3, "tempo": 0.2},
        "angry":    {"mean_shift": 2.0, "energy": 1.0, "pitch_var": 0.9, "tempo": 0.8},
        "fearful":  {"mean_shift": 0.8, "energy": 0.6, "pitch_var": 0.7, "tempo": 0.7},
        "disgust":  {"mean_shift": -0.5, "energy": 0.4, "pitch_var": 0.5, "tempo": 0.4},
    }

    X_all = []
    y_all = []

    for emotion in SELECTED_EMOTIONS:
        label = EMOTION_TO_LABEL[emotion]
        profile = emotion_profiles[emotion]

        # MFCC features (80 = 40 mean + 40 std)
        mfcc_mean = rng.randn(n_per_class, 40) * 0.8 + profile["mean_shift"]
        mfcc_std = np.abs(rng.randn(n_per_class, 40) * 0.3 + profile["pitch_var"])

        # Chroma features (12)
        chroma = rng.randn(n_per_class, 12) * 0.4 + profile["mean_shift"] * 0.5

        # Mel spectrogram features (128)
        mel = rng.randn(n_per_class, 128) * 0.5 + profile["energy"]

        # ZCR (1) and RMS (1)
        zcr = rng.randn(n_per_class, 1) * 0.2 + profile["tempo"]
        rms = rng.randn(n_per_class, 1) * 0.15 + profile["energy"]

        features = np.hstack([mfcc_mean, mfcc_std, chroma, mel, zcr, rms])
        assert features.shape[1] == n_features, (
            f"Feature count mismatch: {features.shape[1]} != {n_features}"
        )

        X_all.append(features)
        y_all.append(np.full(n_per_class, label, dtype=np.int64))

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    # Shuffle the data
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"[DataLoader] Generated {len(X)} synthetic samples ({n_per_class}/class, "
          f"{n_classes} classes, {n_features} features).")

    return X, y, SELECTED_EMOTIONS


if __name__ == "__main__":
    # Quick test
    print("=== Testing Synthetic Data Generation ===")
    X, y, emotions = generate_synthetic_features(n_per_class=50)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Emotions: {emotions}")
    print(f"Label distribution: {np.bincount(y)}")
