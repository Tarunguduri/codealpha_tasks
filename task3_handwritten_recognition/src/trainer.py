"""
Trainer Module
==============
Handles model training with data augmentation and callbacks.
Supports both augmented (ImageDataGenerator) and non-augmented training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, Any, Optional


def get_augmentation_generator() -> ImageDataGenerator:
    """
    Create an ImageDataGenerator with sensible augmentation for handwritten chars.

    Augmentations:
        - Rotation: ±10 degrees
        - Zoom: ±10%
        - Width/Height shift: ±10%

    Returns:
        Configured ImageDataGenerator instance.
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    return datagen


def get_callbacks(patience_early: int = 5, patience_lr: int = 3) -> list:
    """
    Create training callbacks for early stopping and learning rate reduction.

    Args:
        patience_early: Epochs to wait before early stopping (default: 5).
        patience_lr: Epochs to wait before reducing LR (default: 3).

    Returns:
        List of Keras callback instances.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 128,
    use_augmentation: bool = True,
) -> Dict[str, Any]:
    """
    Train a Keras model with optional data augmentation and callbacks.

    Args:
        model: Compiled Keras model to train.
        x_train: Training images (N, 28, 28, 1), normalized.
        y_train: Training labels (N,).
        x_val: Validation images.
        y_val: Validation labels.
        epochs: Maximum training epochs (default: 15).
        batch_size: Training batch size (default: 128).
        use_augmentation: Whether to use ImageDataGenerator (default: True).

    Returns:
        Dictionary containing training history and metadata.
    """
    model_name = model.name
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"  Augmentation: {'ON' if use_augmentation else 'OFF'}")
    print(f"{'='*60}\n")

    callbacks = get_callbacks()

    if use_augmentation:
        datagen = get_augmentation_generator()
        datagen.fit(x_train)

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

    # Extract history
    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }

    result = {
        "model_name": model_name,
        "epochs_trained": len(history.history["accuracy"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "use_augmentation": use_augmentation,
        "batch_size": batch_size,
        "history": history_dict,
    }

    print(f"\n[Trainer] {model_name} training complete!")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"  Final train accuracy: {result['final_train_accuracy']:.4f}")

    return result


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
    print("Augmentation generator settings:")
    gen = get_augmentation_generator()
    print(f"  Rotation range: {gen.rotation_range}")
    print(f"  Zoom range: {gen.zoom_range}")
    print(f"  Width shift range: {gen.width_shift_range}")
    print(f"  Height shift range: {gen.height_shift_range}")
