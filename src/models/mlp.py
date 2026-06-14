from __future__ import annotations

from pathlib import Path

import numpy as np


class Flatten:
    def __init__(self):
        self.cache_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.cache_shape)


class Dense:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.mask = np.ones_like(self.weights, dtype=np.float32)
        self.cache_input: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_input is not None
        self.grad_weights = self.cache_input.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weights.T

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.weights *= self.mask
        self.bias -= learning_rate * self.grad_bias

    def prune_by_threshold(self, threshold: float) -> None:
        self.mask = (np.abs(self.weights) > threshold).astype(np.float32)
        self.weights *= self.mask

    @property
    def total_parameters(self) -> int:
        return int(self.weights.size)

    @property
    def zero_parameters(self) -> int:
        return int(np.sum(self.mask == 0.0))


class ReLU:
    def __init__(self):
        self.cache_input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return np.maximum(x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_input is not None
        return grad_output * (self.cache_input > 0.0)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probabilities: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.labels = labels
        correct_log_probs = -np.log(self.probabilities[np.arange(len(labels)), labels] + 1e-12)
        return float(np.mean(correct_log_probs))

    def backward(self) -> np.ndarray:
        assert self.probabilities is not None and self.labels is not None
        batch_size = len(self.labels)
        grad = self.probabilities.copy()
        grad[np.arange(batch_size), self.labels] -= 1.0
        return grad / batch_size


class ManualMLP:
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        hidden_units: int = 64,
        dense_layers: int = 2,
    ):
        self.input_length = input_length
        self.flatten = Flatten()
        self.layers: list[object] = []
        current_features = input_length

        for index in range(dense_layers):
            self.layers.append(Dense(current_features, hidden_units))
            self.layers.append(ReLU())
            current_features = hidden_units

        self.output_layer = Dense(current_features, num_classes)
        self.loss = SoftmaxCrossEntropy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.flatten.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        logits = self.output_layer.forward(x)
        return logits

    def backward(self, grad_output: np.ndarray) -> None:
        grad = self.output_layer.backward(grad_output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        grad = self.flatten.backward(grad)
        return grad

    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update(learning_rate)
        self.output_layer.update(learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def get_parameters(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                params[f"dense_{layer_index}.weights"] = layer.weights
                params[f"dense_{layer_index}.bias"] = layer.bias
                params[f"dense_{layer_index}.mask"] = layer.mask
        params["output_layer.weights"] = self.output_layer.weights
        params["output_layer.bias"] = self.output_layer.bias
        params["output_layer.mask"] = self.output_layer.mask
        return params

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.weights = params[f"dense_{layer_index}.weights"].astype(np.float32)
                layer.bias = params[f"dense_{layer_index}.bias"].astype(np.float32)
                layer.mask = params.get(f"dense_{layer_index}.mask", np.ones_like(layer.weights)).astype(np.float32)
        self.output_layer.weights = params["output_layer.weights"].astype(np.float32)
        self.output_layer.bias = params["output_layer.bias"].astype(np.float32)
        self.output_layer.mask = params.get("output_layer.mask", np.ones_like(self.output_layer.weights)).astype(np.float32)

    def prune_by_fraction(self, prune_fraction: float) -> None:
        if prune_fraction <= 0.0:
            return
        if prune_fraction >= 1.0:
            for layer in self._all_dense_layers():
                layer.mask = np.zeros_like(layer.weights)
                layer.weights *= layer.mask
            return

        weights = np.concatenate(
            [np.abs(layer.weights).flatten() for layer in self._all_dense_layers()]
        )
        threshold = np.percentile(weights, prune_fraction * 100)
        for layer in self._all_dense_layers():
            layer.prune_by_threshold(threshold)

    def _all_dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)] + [self.output_layer]

    def total_parameters(self) -> int:
        return sum(layer.total_parameters for layer in self._all_dense_layers())

    def sparsity(self) -> float:
        zeros = sum(layer.zero_parameters for layer in self._all_dense_layers())
        return float(zeros / self.total_parameters())

    def save_keras_model(self, path: Path) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to export the manual MLP to Keras.") from exc

        inputs = tf.keras.Input(shape=(self.input_length, 1), name="ecg_input")
        x = tf.keras.layers.Flatten(name="flatten")(inputs)
        dense_index = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                x = tf.keras.layers.Dense(
                    layer.out_features,
                    activation="linear",
                    name=f"dense_{dense_index}",
                )(x)
                dense_index += 1
            elif isinstance(layer, ReLU):
                x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Dense(self.output_layer.out_features, activation="softmax", name="output_layer")(x)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="manual_mlp_export")

        keras_model(np.zeros((1, self.input_length, 1), dtype=np.float32))

        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                keras_layer = keras_model.get_layer(name=f"dense_{dense_index}")
                keras_layer.set_weights([layer.weights, layer.bias])
                dense_index += 1
        keras_model.get_layer(name="output_layer").set_weights([self.output_layer.weights, self.output_layer.bias])
        keras_model.save(path)


def build_manual_mlp(
    input_length: int,
    num_classes: int,
    hidden_units: int = 64,
    dense_layers: int = 2,
) -> ManualMLP:
    return ManualMLP(
        input_length=input_length,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dense_layers=dense_layers,
    )
