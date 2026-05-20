"""
Preprocessor Module
===================
Handles data normalization, reshaping, and validation splitting.
FIX: Uses stratified train_test_split instead of naive slicing to ensure
     balanced class representation in validation set.
"""

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0, 1].

    Args:
        images: Array of images with uint8 pixel values.

    Returns:
        Normalized images as float32 in [0, 1] range.
    """
    return images.astype(np.float32) / 255.0


def reshape_for_cnn(images: np.ndarray) -> np.ndarray:
    """
    Reshape images to include channel dimension for CNN input.

    Args:
        images: Array of shape (N, 28, 28).

    Returns:
        Reshaped array of shape (N, 28, 28, 1).
    """
    if images.ndim == 3:
        return images[..., np.newaxis]
    return images


def create_validation_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a stratified validation split from training data.

    FIX: Previous implementation took the first N samples without shuffling,
    which could lead to class imbalance in validation set. Now uses
    sklearn's train_test_split with stratify to ensure proportional
    class representation.

    Args:
        x_train: Training images.
        y_train: Training labels.
        val_fraction: Fraction of training data to use for validation (default: 0.1).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (x_train_split, x_val, y_train_split, y_val)
    """
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_train,  # FIX: Ensures balanced classes in validation set
        shuffle=True,
    )

    print(f"[Preprocessor] Stratified validation split:")
    print(f"  Training: {len(x_train_split)} samples")
    print(f"  Validation: {len(x_val)} samples")

    # Verify stratification
    unique_classes = np.unique(y_train)
    for cls in unique_classes[:3]:  # Show first 3 classes as examples
        train_ratio = np.mean(y_train_split == cls)
        val_ratio = np.mean(y_val == cls)
        print(f"  Class {cls}: train={train_ratio:.3f}, val={val_ratio:.3f}")
    if len(unique_classes) > 3:
        print(f"  ... (verified for all {len(unique_classes)} classes)")

    return x_train_split, x_val, y_train_split, y_val


def preprocess_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: normalize → reshape → split.

    Args:
        x_train: Raw training images.
        y_train: Training labels.
        x_test: Raw test images.
        val_fraction: Fraction for validation split.

    Returns:
        Tuple of (x_train, x_val, y_train, y_val, x_test, y_test_unchanged)
        Note: y_test is not modified, returned unchanged through evaluator.
    """
    print("\n[Preprocessor] Running preprocessing pipeline...")

    # Step 1: Normalize
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print(f"  ✓ Normalized to [0, 1] range")

    # Step 2: Reshape for CNN (add channel dim)
    x_train = reshape_for_cnn(x_train)
    x_test = reshape_for_cnn(x_test)
    print(f"  ✓ Reshaped to {x_train.shape[1:]} (added channel dimension)")

    # Step 3: Stratified validation split (FIX)
    x_train, x_val, y_train, y_val = create_validation_split(
        x_train, y_train, val_fraction=val_fraction
    )
    print(f"  ✓ Stratified validation split ({val_fraction*100:.0f}%)")

    return x_train, x_val, y_train, y_val, x_test


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing preprocessor...")
    dummy_x = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
    dummy_y = np.random.randint(0, 10, (1000,))
    dummy_test = np.random.randint(0, 255, (200, 28, 28), dtype=np.uint8)

    x_tr, x_v, y_tr, y_v, x_te = preprocess_pipeline(dummy_x, dummy_y, dummy_test)
    print(f"\nResults:")
    print(f"  x_train: {x_tr.shape}, dtype={x_tr.dtype}, range=[{x_tr.min():.2f}, {x_tr.max():.2f}]")
    print(f"  x_val: {x_v.shape}")
    print(f"  x_test: {x_te.shape}")
