"""
Models Module
=============
Defines model architectures for emotion recognition:
1. CNN (1D Convolutional Neural Network)
2. Bi-LSTM (Bidirectional LSTM)
3. MLP (sklearn MLPClassifier as TensorFlow fallback)

Gracefully falls back to sklearn if TensorFlow is not installed.
"""

import numpy as np
from typing import Optional, Tuple

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
    print(f"[Models] TensorFlow {tf.__version__} available.")
except ImportError:
    TF_AVAILABLE = False
    print("[Models] TensorFlow not installed. Using sklearn MLP fallback.")

# sklearn is always available
from sklearn.neural_network import MLPClassifier


def build_cnn(
    input_shape: Tuple[int, ...],
    n_classes: int,
    learning_rate: float = 1e-3,
) -> object:
    """
    Build a 1D CNN for emotion classification.

    Architecture:
        Conv1D(64, k=5) → BatchNorm → MaxPool → Dropout(0.3)
        Conv1D(128, k=5) → BatchNorm → MaxPool → Dropout(0.3)
        Conv1D(256, k=3) → GlobalAveragePooling → Dropout(0.4)
        Dense(128, relu) → Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (n_features, 1) for 1D conv.
    n_classes : int
        Number of emotion classes.
    learning_rate : float
        Learning rate for Adam optimizer.

    Returns
    -------
    model : keras.Model
        Compiled Keras model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Use build_mlp() instead.")

    model = keras.Sequential([
        # Block 1
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same",
                       input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 2
        layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),

        # Classifier
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="EmotionCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def build_lstm(
    input_shape: Tuple[int, ...],
    n_classes: int,
    learning_rate: float = 1e-3,
) -> object:
    """
    Build a Bidirectional LSTM for emotion classification.

    Architecture:
        Bi-LSTM(128, return_sequences=True) → Dropout(0.3)
        Bi-LSTM(64) → Dropout(0.3)
        Dense(64, relu) → Dropout(0.3)
        Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (timesteps, features).
    n_classes : int
        Number of emotion classes.
    learning_rate : float
        Learning rate for Adam optimizer.

    Returns
    -------
    model : keras.Model
        Compiled Keras model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Use build_mlp() instead.")

    model = keras.Sequential([
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="EmotionBiLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def build_mlp(
    n_classes: int,
    hidden_layers: Tuple[int, ...] = (256, 128, 64),
    max_iter: int = 500,
    random_state: int = 42,
) -> MLPClassifier:
    """
    Build a sklearn MLPClassifier as a TensorFlow fallback.

    Parameters
    ----------
    n_classes : int
        Number of emotion classes (used for info only, sklearn infers from data).
    hidden_layers : tuple of int
        Hidden layer sizes.
    max_iter : int
        Maximum training iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : MLPClassifier
        Configured sklearn MLP classifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=random_state,
        verbose=True,
    )

    print(f"[Models] Built sklearn MLP: layers={hidden_layers}, max_iter={max_iter}")
    return model


def is_tf_available() -> bool:
    """Check if TensorFlow is available."""
    return TF_AVAILABLE


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Reshape 2D features to 3D for Conv1D: (samples, features, 1)."""
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_for_lstm(X: np.ndarray, timesteps: int = 1) -> np.ndarray:
    """Reshape 2D features to 3D for LSTM: (samples, timesteps, features)."""
    return X.reshape(X.shape[0], timesteps, X.shape[1])


if __name__ == "__main__":
    print(f"TensorFlow available: {TF_AVAILABLE}")
    n_classes = 6
    n_features = 222

    if TF_AVAILABLE:
        cnn = build_cnn((n_features, 1), n_classes)
        lstm = build_lstm((1, n_features), n_classes)

    mlp = build_mlp(n_classes)
    print("All models created successfully.")
