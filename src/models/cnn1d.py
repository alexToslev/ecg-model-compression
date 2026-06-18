from __future__ import annotations

import tensorflow as tf


def build_improved_cnn(
    input_length: int,
    num_classes: int,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Build a compact 1D CNN for MIT-BIH heartbeat classification.

    The model is intentionally small enough for later TensorFlow Lite export:
    three Conv1D blocks extract local ECG waveform features, GlobalAveragePooling
    keeps the dense part compact, and Dropout reduces overfitting after
    oversampling rare classes.
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="ecg_input")

    x = tf.keras.layers.Conv1D(16, kernel_size=7, padding="same", use_bias=False, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool1")(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool2")(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(x)

    x = tf.keras.layers.Dense(32, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="improved_ecg_cnn")


def build_tiny_cnn(input_length: int, num_classes: int) -> tf.keras.Model:
    """Backward-compatible alias used by older scripts."""
    return build_improved_cnn(input_length=input_length, num_classes=num_classes)
