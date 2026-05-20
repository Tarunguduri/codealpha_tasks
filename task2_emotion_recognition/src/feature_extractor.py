"""
Feature Extractor Module
========================
Extracts acoustic features from audio files for emotion recognition:
- 40 MFCCs (mean + std = 80 features)
- 12 Chroma features (mean)
- 128 Mel spectrogram bands (mean)
- 1 Zero Crossing Rate (mean)
- 1 RMS Energy (mean)
Total: 222 features per sample

Gracefully falls back if librosa is not installed.
"""

import numpy as np
from typing import Optional, List
from tqdm import tqdm

# Try importing librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[FeatureExtractor] librosa not installed. Audio feature extraction disabled.")

# Feature configuration
N_MFCC: int = 40
N_CHROMA: int = 12
N_MEL: int = 128
SAMPLE_RATE: int = 22050
DURATION: float = 3.0  # seconds to load from each clip
N_FEATURES: int = N_MFCC * 2 + N_CHROMA + N_MEL + 1 + 1  # 222


def extract_features_from_file(
    file_path: str,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> Optional[np.ndarray]:
    """
    Extract acoustic features from a single audio file.

    Parameters
    ----------
    file_path : str
        Path to the .wav audio file.
    sr : int
        Target sample rate for loading.
    duration : float
        Maximum duration (seconds) to load from the file.

    Returns
    -------
    features : np.ndarray of shape (222,) or None
        Extracted feature vector, or None on failure.
    """
    if not LIBROSA_AVAILABLE:
        return None

    try:
        # Load audio with fixed duration for consistency
        y, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Pad short clips to ensure consistent feature dimensions
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")

        # --- MFCC (40 coefficients → mean + std = 80) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)   # (40,)
        mfcc_std = np.std(mfcc, axis=1)      # (40,)

        # --- Chroma (12 pitch classes → mean = 12) ---
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=N_CHROMA)
        chroma_mean = np.mean(chroma, axis=1)  # (12,)

        # --- Mel Spectrogram (128 bands → mean = 128) ---
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
        mel_mean = np.mean(mel, axis=1)  # (128,)

        # --- Zero Crossing Rate (mean = 1) ---
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)  # scalar

        # --- RMS Energy (mean = 1) ---
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)  # scalar

        # Concatenate all features
        features = np.concatenate([
            mfcc_mean,     # 40
            mfcc_std,      # 40
            chroma_mean,   # 12
            mel_mean,      # 128
            [zcr_mean],    # 1
            [rms_mean],    # 1
        ])  # Total: 222

        assert features.shape[0] == N_FEATURES, (
            f"Feature count mismatch: {features.shape[0]} != {N_FEATURES}"
        )

        return features

    except Exception as e:
        print(f"[FeatureExtractor] Error processing {file_path}: {e}")
        return None


def extract_features_batch(
    file_paths: np.ndarray,
    labels: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> tuple:
    """
    Extract features from a batch of audio files.

    Parameters
    ----------
    file_paths : np.ndarray
        Array of file paths to process.
    labels : np.ndarray
        Corresponding labels.
    sr : int
        Target sample rate.
    duration : float
        Max duration per clip.

    Returns
    -------
    X : np.ndarray, shape (n_valid, 222)
        Feature matrix for successfully processed files.
    y : np.ndarray, shape (n_valid,)
        Labels for successfully processed files.
    """
    if not LIBROSA_AVAILABLE:
        print("[FeatureExtractor] librosa not available. Cannot extract features from audio.")
        return None, None

    features_list = []
    labels_list = []
    failed = 0

    print(f"[FeatureExtractor] Extracting features from {len(file_paths)} audio files...")

    for fpath, label in tqdm(zip(file_paths, labels), total=len(file_paths),
                              desc="Extracting features"):
        feat = extract_features_from_file(fpath, sr=sr, duration=duration)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(label)
        else:
            failed += 1

    if failed > 0:
        print(f"[FeatureExtractor] Warning: {failed}/{len(file_paths)} files failed extraction.")

    if not features_list:
        return None, None

    X = np.vstack(features_list)
    y = np.array(labels_list, dtype=np.int64)

    print(f"[FeatureExtractor] Successfully extracted features: {X.shape}")
    return X, y


def is_librosa_available() -> bool:
    """Check if librosa is available for audio processing."""
    return LIBROSA_AVAILABLE


if __name__ == "__main__":
    print(f"librosa available: {LIBROSA_AVAILABLE}")
    print(f"Expected feature count: {N_FEATURES}")
    print(f"  MFCC mean: {N_MFCC}, MFCC std: {N_MFCC}")
    print(f"  Chroma: {N_CHROMA}")
    print(f"  Mel: {N_MEL}")
    print(f"  ZCR: 1, RMS: 1")
    print(f"  Total: {N_MFCC * 2 + N_CHROMA + N_MEL + 2}")
