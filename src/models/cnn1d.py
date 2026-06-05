from __future__ import annotations

import tensorflow as tf


def build_baseline_cnn(
    input_length: int,
    num_classes: int,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_length, 1), name="ecg")

    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_ecg_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
