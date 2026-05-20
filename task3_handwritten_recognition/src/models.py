"""
Models Module
=============
CNN architectures for handwritten character recognition.
Provides SimpleCNN and DeepCNN with proper regularization.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_simple_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    n_classes: int = 10,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a simple CNN architecture for character recognition.

    Architecture:
        Conv2D(32, 3×3) → BatchNorm → MaxPool → Dropout(0.25)
        Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout(0.25)
        Flatten → Dense(128) → Dropout(0.4) → Dense(n_classes, softmax)

    Args:
        input_shape: Input image dimensions (default: 28×28×1).
        n_classes: Number of output classes (default: 10 for digits).
        learning_rate: Adam optimizer learning rate (default: 1e-3).

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="SimpleCNN")

    # Block 1: Conv → BN → Pool → Dropout
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2: Conv → BN → Pool → Dropout
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(n_classes, activation="softmax"))

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[Models] SimpleCNN built: {model.count_params():,} parameters")
    return model


def build_deep_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    n_classes: int = 10,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build a deeper CNN architecture with more convolutional layers.

    Architecture:
        Block1: 2×Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
        Block2: 2×Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
        Block3: Conv2D(128) → BatchNorm → GlobalAveragePooling → Dropout(0.4)
        Dense(256) → Dense(n_classes, softmax)

    Args:
        input_shape: Input image dimensions (default: 28×28×1).
        n_classes: Number of output classes (default: 10 for digits).
        learning_rate: Adam optimizer learning rate (default: 1e-3).

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="DeepCNN")

    # Block 1: 2×Conv2D(32) → BN → Pool → Dropout
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2: 2×Conv2D(64) → BN → Pool → Dropout
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3: Conv2D(128) → BN → GAP → Dropout
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))

    # Classifier head
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[Models] DeepCNN built: {model.count_params():,} parameters")
    return model


def get_model_summary_dict(model: tf.keras.Model) -> dict:
    """Extract model architecture info as a dictionary."""
    layer_info = []
    for layer in model.layers:
        info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A",
            "params": layer.count_params(),
        }
        layer_info.append(info)

    return {
        "model_name": model.name,
        "total_params": model.count_params(),
        "trainable_params": sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        ),
        "layers": layer_info,
        "n_layers": len(model.layers),
    }


if __name__ == "__main__":
    print("Building SimpleCNN...")
    simple = build_simple_cnn(n_classes=10)
    simple.summary()

    print("\n" + "="*60)

    print("\nBuilding DeepCNN...")
    deep = build_deep_cnn(n_classes=10)
    deep.summary()
