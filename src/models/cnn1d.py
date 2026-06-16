from __future__ import annotations

from pathlib import Path

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    return np.eye(num_classes, dtype=np.float32)[labels]


def fake_quantize_tensor(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    max_val = float(np.max(np.abs(tensor)))
    scale = max(max_val / 127.0, 1e-8)
    quantized = np.round(tensor / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def fake_quantize_activation(x: np.ndarray) -> np.ndarray:
    quantized, scale = fake_quantize_tensor(x)
    return quantized.astype(np.float32) * scale


class Conv1D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size
        ).astype(np.float32) * np.sqrt(2.0 / (in_channels * kernel_size))
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache_input = None
        self.cache_padded = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        batch_size, length, channels = x.shape
        assert channels == self.in_channels, "Conv1D expects the configured number of input channels."

        if self.padding == "same":
            pad = (self.kernel_size - 1) // 2
            x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        else:
            x_padded = x

        self.cache_padded = x_padded
        out_length = (x_padded.shape[1] - self.kernel_size) // self.stride + 1
        patches = np.stack(
            [x_padded[:, i : i + self.kernel_size, :] for i in range(0, out_length * self.stride, self.stride)],
            axis=1,
        )
        # patches shape: (N, out_length, kernel_size, in_channels)

        output = np.einsum("nlkc,fck->nlf", patches, self.weights) + self.bias[np.newaxis, np.newaxis, :]
        self.cache_patches = patches
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size, out_length, out_channels = grad_output.shape

        self.grad_bias = grad_output.sum(axis=(0, 1))
        self.grad_weights = np.einsum("nlkc,nlf->fck", self.cache_patches, grad_output)

        grad_patches = np.einsum("nlf,fck->nlkc", grad_output, self.weights)
        grad_input_padded = np.zeros_like(self.cache_padded, dtype=np.float32)

        for step in range(out_length):
            start = step * self.stride
            grad_input_padded[:, start : start + self.kernel_size, :] += grad_patches[:, step, :, :]

        if self.padding == "same":
            pad = (self.kernel_size - 1) // 2
            return grad_input_padded[:, pad:-pad, :]
        return grad_input_padded

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

    def fake_quantize_weights(self) -> float:
        quantized, scale = fake_quantize_tensor(self.weights)
        self.weights = quantized.astype(np.float32) * scale
        return scale

    def prune_by_threshold(self, threshold: float) -> None:
        self.weights[np.abs(self.weights) <= threshold] = 0.0
        self.bias[np.abs(self.bias) <= threshold] = 0.0

    def prune_filters_by_fraction(self, prune_fraction: float) -> np.ndarray:
        keep_filters = np.ones(self.out_channels, dtype=np.float32)
        if prune_fraction <= 0.0:
            return keep_filters
        if prune_fraction >= 1.0:
            self.weights.fill(0.0)
            self.bias.fill(0.0)
            return np.zeros(self.out_channels, dtype=np.float32)

        filter_norms = np.linalg.norm(self.weights.reshape(self.out_channels, -1), axis=1)
        threshold = np.percentile(filter_norms, prune_fraction * 100)
        keep_filters = filter_norms > threshold
        if not np.any(keep_filters):
            keep_filters[np.argmax(filter_norms)] = True

        self.weights[~keep_filters, :, :] = 0.0
        self.bias[~keep_filters] = 0.0
        return keep_filters.astype(np.float32)

    def prune_input_channels(self, channel_mask: np.ndarray) -> None:
        if channel_mask.shape[0] != self.in_channels:
            raise ValueError(
                f"Channel mask length {channel_mask.shape[0]} does not match Conv1D input channels {self.in_channels}."
            )
        self.weights[:, channel_mask == 0.0, :] = 0.0

    @property
    def active_filters(self) -> int:
        active_weights = np.any(self.weights != 0.0, axis=(1, 2))
        active_bias = self.bias != 0.0
        return int(np.sum(active_weights | active_bias))

    @property
    def total_parameters(self) -> int:
        return int(self.weights.size + self.bias.size)

    @property
    def zero_parameters(self) -> int:
        return int(np.sum(self.weights == 0.0) + np.sum(self.bias == 0.0))


class ReLU:
    def __init__(self):
        self.cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return np.maximum(x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.cache_input > 0.0)


class MaxPool1D:
    def __init__(self, pool_size: int = 2, stride: int | None = None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.cache_input = None
        self.cache_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        batch_size, length, channels = x.shape
        out_length = length // self.stride
        output = np.zeros((batch_size, out_length, channels), dtype=np.float32)
        self.cache_indices = np.zeros((batch_size, out_length, channels), dtype=np.int32)

        for step in range(out_length):
            start = step * self.stride
            window = x[:, start : start + self.pool_size, :]
            indices = np.argmax(window, axis=1)
            self.cache_indices[:, step, :] = indices
            output[:, step, :] = np.take_along_axis(window, indices[:, np.newaxis, :], axis=1).squeeze(axis=1)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size, out_length, channels = grad_output.shape
        grad_input = np.zeros_like(self.cache_input, dtype=np.float32)

        for step in range(out_length):
            start = step * self.stride
            for batch in range(batch_size):
                for channel in range(channels):
                    index = self.cache_indices[batch, step, channel]
                    grad_input[batch, start + index, channel] = grad_output[batch, step, channel]

        return grad_input


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
        self.cache_input = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weights = self.cache_input.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weights.T

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

    def fake_quantize_weights(self) -> float:
        quantized, scale = fake_quantize_tensor(self.weights)
        self.weights = quantized.astype(np.float32) * scale
        return scale

    def prune_by_threshold(self, threshold: float) -> None:
        self.weights[np.abs(self.weights) <= threshold] = 0.0
        self.bias[np.abs(self.bias) <= threshold] = 0.0

    def prune_neurons_by_fraction(self, prune_fraction: float) -> np.ndarray:
        keep_neurons = np.ones(self.out_features, dtype=np.float32)
        if prune_fraction <= 0.0:
            return keep_neurons
        if prune_fraction >= 1.0:
            self.weights.fill(0.0)
            self.bias.fill(0.0)
            return np.zeros(self.out_features, dtype=np.float32)

        neuron_norms = np.linalg.norm(self.weights, axis=0)
        threshold = np.percentile(neuron_norms, prune_fraction * 100)
        keep_neurons = neuron_norms > threshold
        if not np.any(keep_neurons):
            keep_neurons[np.argmax(neuron_norms)] = True

        self.weights[:, ~keep_neurons] = 0.0
        self.bias[~keep_neurons] = 0.0
        return keep_neurons.astype(np.float32)

    def prune_input_features(self, feature_mask: np.ndarray) -> None:
        if feature_mask.shape[0] != self.in_features:
            raise ValueError(
                f"Feature mask length {feature_mask.shape[0]} does not match Dense input features {self.in_features}."
            )
        self.weights[feature_mask == 0.0, :] = 0.0

    @property
    def active_output_neurons(self) -> int:
        active_weights = np.any(self.weights != 0.0, axis=0)
        active_bias = self.bias != 0.0
        return int(np.sum(active_weights | active_bias))

    @property
    def total_parameters(self) -> int:
        return int(self.weights.size + self.bias.size)

    @property
    def zero_parameters(self) -> int:
        return int(np.sum(self.weights == 0.0) + np.sum(self.bias == 0.0))


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probabilities = None
        self.labels = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.labels = labels
        correct_log_probs = -np.log(self.probabilities[np.arange(len(labels)), labels] + 1e-12)
        return float(np.mean(correct_log_probs))

    def backward(self) -> np.ndarray:
        batch_size = len(self.labels)
        grad = self.probabilities.copy()
        grad[np.arange(batch_size), self.labels] -= 1.0
        return grad / batch_size


class CNNFromScratch:
    def __init__(self, input_length: int, num_classes: int):
        self.input_length = input_length
        self.conv1 = Conv1D(in_channels=1, out_channels=8, kernel_size=7, padding="same")
        self.relu1 = ReLU()
        self.pool1 = MaxPool1D(pool_size=2)

        self.conv2 = Conv1D(in_channels=8, out_channels=16, kernel_size=5, padding="same")
        self.relu2 = ReLU()
        self.pool2 = MaxPool1D(pool_size=2)

        self.conv3 = Conv1D(in_channels=16, out_channels=32, kernel_size=3, padding="same")
        self.relu3 = ReLU()
        self.pool3 = MaxPool1D(pool_size=2)

        self.flatten = Flatten()
        flattened_size = (input_length // 8) * 32
        self.dense1 = Dense(flattened_size, 64)
        self.relu4 = ReLU()
        self.output_layer = Dense(64, num_classes)
        self.loss = SoftmaxCrossEntropy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)

        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.relu4.forward(x)
        logits = self.output_layer.forward(x)
        return logits

    def forward_quantized(self, x: np.ndarray) -> np.ndarray:
        x = fake_quantize_activation(x)

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = fake_quantize_activation(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = fake_quantize_activation(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = fake_quantize_activation(x)
        x = self.pool3.forward(x)

        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.relu4.forward(x)
        x = fake_quantize_activation(x)
        logits = self.output_layer.forward(x)
        return logits

    def backward(self, grad_output: np.ndarray) -> None:
        grad = self.output_layer.backward(grad_output)
        grad = self.relu4.backward(grad)
        grad = self.dense1.backward(grad)
        grad = self.flatten.backward(grad)

        grad = self.pool3.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def update(self, learning_rate: float) -> None:
        self.conv1.update(learning_rate)
        self.conv2.update(learning_rate)
        self.conv3.update(learning_rate)
        self.dense1.update(learning_rate)
        self.output_layer.update(learning_rate)

    def fake_quantize_weights(self) -> None:
        self.conv1.fake_quantize_weights()
        self.conv2.fake_quantize_weights()
        self.conv3.fake_quantize_weights()
        self.dense1.fake_quantize_weights()
        self.output_layer.fake_quantize_weights()

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def save(self, path: str | Path) -> None:
        self.save_keras_model(path)

    def save_keras_model(self, path: str | Path) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to export a Keras model from the scratch CNN."
            ) from exc

        path = Path(path)
        inputs = tf.keras.Input(shape=(self.input_length, 1), name="ecg_input")
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, padding="same", activation=None, name="conv1")(inputs)
        x = tf.keras.layers.ReLU(name="relu1")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool1")(x)

        x = tf.keras.layers.Conv1D(filters=16, kernel_size=5, padding="same", activation=None, name="conv2")(x)
        x = tf.keras.layers.ReLU(name="relu2")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool2")(x)

        x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation=None, name="conv3")(x)
        x = tf.keras.layers.ReLU(name="relu3")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool3")(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(64, activation=None, name="dense1")(x)
        x = tf.keras.layers.ReLU(name="relu4")(x)
        outputs = tf.keras.layers.Dense(self.output_layer.out_features, activation="softmax", name="output_layer")(x)

        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_ecg_cnn")
        keras_model.get_layer("conv1").set_weights([
            self.conv1.weights.transpose(2, 1, 0),
            self.conv1.bias,
        ])
        keras_model.get_layer("conv2").set_weights([
            self.conv2.weights.transpose(2, 1, 0),
            self.conv2.bias,
        ])
        keras_model.get_layer("conv3").set_weights([
            self.conv3.weights.transpose(2, 1, 0),
            self.conv3.bias,
        ])
        keras_model.get_layer("dense1").set_weights(
            [self.dense1.weights, self.dense1.bias]
        )
        keras_model.get_layer("output_layer").set_weights(
            [self.output_layer.weights, self.output_layer.bias]
        )

        keras_model.save(path)

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {
            "conv1.weights": self.conv1.weights,
            "conv1.bias": self.conv1.bias,
            "conv2.weights": self.conv2.weights,
            "conv2.bias": self.conv2.bias,
            "conv3.weights": self.conv3.weights,
            "conv3.bias": self.conv3.bias,
            "dense1.weights": self.dense1.weights,
            "dense1.bias": self.dense1.bias,
            "output_layer.weights": self.output_layer.weights,
            "output_layer.bias": self.output_layer.bias,
        }

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.conv1.weights = params["conv1.weights"].astype(np.float32)
        self.conv1.bias = params["conv1.bias"].astype(np.float32)
        self.conv2.weights = params["conv2.weights"].astype(np.float32)
        self.conv2.bias = params["conv2.bias"].astype(np.float32)
        self.conv3.weights = params["conv3.weights"].astype(np.float32)
        self.conv3.bias = params["conv3.bias"].astype(np.float32)
        self.dense1.weights = params["dense1.weights"].astype(np.float32)
        self.dense1.bias = params["dense1.bias"].astype(np.float32)
        self.output_layer.weights = params["output_layer.weights"].astype(np.float32)
        self.output_layer.bias = params["output_layer.bias"].astype(np.float32)

    def prune_structured_by_fraction(self, prune_fraction: float) -> None:
        conv1_mask = self.conv1.prune_filters_by_fraction(prune_fraction)
        self.conv2.prune_input_channels(conv1_mask)

        conv2_mask = self.conv2.prune_filters_by_fraction(prune_fraction)
        self.conv3.prune_input_channels(conv2_mask)

        conv3_mask = self.conv3.prune_filters_by_fraction(prune_fraction)
        pooled_length = self.input_length // 8
        dense_input_mask = np.tile(conv3_mask, pooled_length)
        self.dense1.prune_input_features(dense_input_mask)

        dense1_mask = self.dense1.prune_neurons_by_fraction(prune_fraction)
        self.output_layer.prune_input_features(dense1_mask)

    def prune_magnitude_by_fraction(self, prune_fraction: float) -> None:
        if prune_fraction <= 0.0:
            return
        if prune_fraction >= 1.0:
            for layer in self._weighted_layers():
                layer.weights.fill(0.0)
                layer.bias.fill(0.0)
            return

        weights = np.concatenate([np.abs(layer.weights).ravel() for layer in self._weighted_layers()])
        threshold = np.percentile(weights, prune_fraction * 100)
        for layer in self._weighted_layers():
            layer.prune_by_threshold(threshold)

    def _weighted_layers(self) -> list[Conv1D | Dense]:
        return [self.conv1, self.conv2, self.conv3, self.dense1, self.output_layer]

    def structured_counts(self) -> dict[str, int]:
        total_filters = self.conv1.out_channels + self.conv2.out_channels + self.conv3.out_channels
        active_filters = self.conv1.active_filters + self.conv2.active_filters + self.conv3.active_filters
        total_hidden_neurons = self.dense1.out_features
        active_hidden_neurons = self.dense1.active_output_neurons
        return {
            "total_conv_filters": int(total_filters),
            "active_conv_filters": int(active_filters),
            "pruned_conv_filters": int(total_filters - active_filters),
            "total_hidden_neurons": int(total_hidden_neurons),
            "active_hidden_neurons": int(active_hidden_neurons),
            "pruned_hidden_neurons": int(total_hidden_neurons - active_hidden_neurons),
        }

    def structured_sparsity(self) -> float:
        counts = self.structured_counts()
        total_structures = counts["total_conv_filters"] + counts["total_hidden_neurons"]
        pruned_structures = counts["pruned_conv_filters"] + counts["pruned_hidden_neurons"]
        if total_structures == 0:
            return 0.0
        return float(pruned_structures / total_structures)

    def total_parameters(self) -> int:
        return (
            self.conv1.total_parameters
            + self.conv2.total_parameters
            + self.conv3.total_parameters
            + self.dense1.total_parameters
            + self.output_layer.total_parameters
        )

    def zero_parameters(self) -> int:
        return (
            self.conv1.zero_parameters
            + self.conv2.zero_parameters
            + self.conv3.zero_parameters
            + self.dense1.zero_parameters
            + self.output_layer.zero_parameters
        )

    def sparsity(self) -> float:
        return float(self.zero_parameters() / self.total_parameters())


def build_tiny_cnn(input_length: int, num_classes: int) -> CNNFromScratch:
    return CNNFromScratch(input_length=input_length, num_classes=num_classes)
