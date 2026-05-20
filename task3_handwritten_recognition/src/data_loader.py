"""
Data Loader Module
==================
Handles loading of MNIST and EMNIST datasets for handwritten character recognition.
"""

import numpy as np
from typing import Tuple, Dict, Any


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the MNIST handwritten digits dataset.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test, metadata)
        - x_train: Training images (60000, 28, 28)
        - y_train: Training labels (60000,)
        - x_test: Test images (10000, 28, 28)
        - y_test: Test labels (10000,)
        - metadata: Dict with dataset info (n_classes, class_names, dataset_name)
    """
    from tensorflow.keras.datasets import mnist

    print("[DataLoader] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    class_names = [str(i) for i in range(10)]
    metadata = {
        "dataset_name": "MNIST",
        "n_classes": 10,
        "class_names": class_names,
        "image_shape": (28, 28),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
    }

    print(f"[DataLoader] MNIST loaded: {len(x_train)} train, {len(x_test)} test samples")
    print(f"[DataLoader] Image shape: {x_train.shape[1:]}, Classes: {metadata['n_classes']}")

    return x_train, y_train, x_test, y_test, metadata


def load_emnist_letters() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the EMNIST Letters dataset (26 handwritten letter classes A-Z).

    Falls back to MNIST if the emnist package is not installed.

    Returns:
        Same format as load_mnist().
    """
    try:
        from emnist import extract_training_samples, extract_test_samples

        print("[DataLoader] Loading EMNIST Letters dataset...")
        x_train, y_train = extract_training_samples("letters")
        x_test, y_test = extract_test_samples("letters")

        # EMNIST letters labels are 1-26 (A-Z), shift to 0-25
        y_train = y_train - 1
        y_test = y_test - 1

        class_names = [chr(ord('A') + i) for i in range(26)]
        metadata = {
            "dataset_name": "EMNIST-Letters",
            "n_classes": 26,
            "class_names": class_names,
            "image_shape": (28, 28),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
        }

        print(f"[DataLoader] EMNIST Letters loaded: {len(x_train)} train, {len(x_test)} test")
        print(f"[DataLoader] Classes: {metadata['n_classes']} (A-Z)")

        return x_train, y_train, x_test, y_test, metadata

    except ImportError:
        print("[DataLoader] WARNING: 'emnist' package not installed. Falling back to MNIST.")
        print("[DataLoader] Install with: pip install emnist")
        return load_mnist()


def get_dataset_summary(metadata: Dict[str, Any]) -> str:
    """Generate a formatted summary string for a dataset."""
    summary = (
        f"\n{'='*50}\n"
        f"  Dataset: {metadata['dataset_name']}\n"
        f"  Classes: {metadata['n_classes']}\n"
        f"  Image Shape: {metadata['image_shape']}\n"
        f"  Training Samples: {metadata['train_samples']}\n"
        f"  Test Samples: {metadata['test_samples']}\n"
        f"{'='*50}\n"
    )
    return summary


if __name__ == "__main__":
    # Quick test
    x_train, y_train, x_test, y_test, meta = load_mnist()
    print(get_dataset_summary(meta))
    print(f"Train shape: {x_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test shape: {x_test.shape}, Labels shape: {y_test.shape}")
    print(f"Label range: {y_train.min()} - {y_train.max()}")
