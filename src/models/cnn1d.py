from __future__ import annotations

from pathlib import Path

import numpy as np


# Manual 1D convolution layer.
class Conv1D:
    # Initializes layer parameters and caches.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weights = (np.random.randn(out_channels, in_channels, kernel_size) * scale).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache_padded: np.ndarray | None = None
        self.cache_patches: np.ndarray | None = None

        self._adam_step = 0
        self._mw = np.zeros_like(self.weights)
        self._vw = np.zeros_like(self.weights)
        self._mb = np.zeros_like(self.bias)
        self._vb = np.zeros_like(self.bias)

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.padding == "same":
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
        else:
            x_padded = x

        out_length = x_padded.shape[1] - self.kernel_size + 1
        patches = np.stack(
            [x_padded[:, start : start + self.kernel_size, :] for start in range(out_length)],
            axis=1,
        )

        if training:
            self.cache_padded = x_padded
            self.cache_patches = patches

        return np.einsum("nlkc,fck->nlf", patches, self.weights) + self.bias[None, None, :]

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_padded is not None and self.cache_patches is not None
        out_length = grad_output.shape[1]

        self.grad_bias = grad_output.sum(axis=(0, 1))
        self.grad_weights = np.einsum("nlkc,nlf->fck", self.cache_patches, grad_output)

        grad_patches = np.einsum("nlf,fck->nlkc", grad_output, self.weights)
        grad_input_padded = np.zeros_like(self.cache_padded, dtype=np.float32)

        for step in range(out_length):
            grad_input_padded[:, step : step + self.kernel_size, :] += grad_patches[:, step, :, :]

        if self.padding == "same":
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            if pad_right == 0:
                return grad_input_padded[:, pad_left:, :]
            return grad_input_padded[:, pad_left:-pad_right, :]
        return grad_input_padded

    # Updates trainable weights.
    def update(self, learning_rate: float, optimizer: str = "sgd") -> None:
        if optimizer == "adam":
            self._adam_step += 1
            self.weights = adam_update(self.weights, self.grad_weights, self._mw, self._vw, self._adam_step, learning_rate)
            self.bias = adam_update(self.bias, self.grad_bias, self._mb, self._vb, self._adam_step, learning_rate)
            return

        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


# ReLU activation layer.
class ReLU:
    # Initializes layer parameters and caches.
    def __init__(self):
        self.cache_input: np.ndarray | None = None

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.cache_input = x
        return np.maximum(x, 0.0)

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_input is not None
        return grad_output * (self.cache_input > 0.0)


# Max-pooling layer for short ECG windows.
class MaxPool1D:
    # Initializes layer parameters and caches.
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.cache_input: np.ndarray | None = None
        self.cache_indices: np.ndarray | None = None

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, length, channels = x.shape
        out_length = length // self.pool_size
        output = np.zeros((batch_size, out_length, channels), dtype=np.float32)
        indices = np.zeros((batch_size, out_length, channels), dtype=np.int32)

        for step in range(out_length):
            start = step * self.pool_size
            window = x[:, start : start + self.pool_size, :]
            best = np.argmax(window, axis=1)
            indices[:, step, :] = best
            output[:, step, :] = np.take_along_axis(window, best[:, None, :], axis=1).squeeze(axis=1)

        if training:
            self.cache_input = x
            self.cache_indices = indices
        return output

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_input is not None and self.cache_indices is not None
        batch_size, out_length, channels = grad_output.shape
        grad_input = np.zeros_like(self.cache_input, dtype=np.float32)

        for step in range(out_length):
            start = step * self.pool_size
            for batch in range(batch_size):
                for channel in range(channels):
                    index = self.cache_indices[batch, step, channel]
                    grad_input[batch, start + index, channel] += grad_output[batch, step, channel]
        return grad_input


# Averages each feature channel over time.
class GlobalAveragePool1D:
    # Initializes layer parameters and caches.
    def __init__(self):
        self.cache_length: int | None = None

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.cache_length = x.shape[1]
        return x.mean(axis=1)

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_length is not None
        return np.repeat(grad_output[:, None, :] / self.cache_length, self.cache_length, axis=1)


# Fully connected layer.
class Dense:
    # Initializes layer parameters and caches.
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = (np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache_input: np.ndarray | None = None

        self._adam_step = 0
        self._mw = np.zeros_like(self.weights)
        self._vw = np.zeros_like(self.weights)
        self._mb = np.zeros_like(self.bias)
        self._vb = np.zeros_like(self.bias)

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.cache_input = x
        return x @ self.weights + self.bias

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        assert self.cache_input is not None
        self.grad_weights = self.cache_input.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weights.T

    # Updates trainable weights.
    def update(self, learning_rate: float, optimizer: str = "sgd") -> None:
        if optimizer == "adam":
            self._adam_step += 1
            self.weights = adam_update(self.weights, self.grad_weights, self._mw, self._vw, self._adam_step, learning_rate)
            self.bias = adam_update(self.bias, self.grad_bias, self._mb, self._vb, self._adam_step, learning_rate)
            return

        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


# Randomly drops hidden features during training.
class Dropout:
    # Initializes layer parameters and caches.
    def __init__(self, rate: float):
        self.rate = rate
        self.mask: np.ndarray | None = None

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate <= 0.0:
            return x
        keep_probability = 1.0 - self.rate
        self.mask = (np.random.random(x.shape) < keep_probability).astype(np.float32) / keep_probability
        return x * self.mask

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.rate <= 0.0:
            return grad_output
        assert self.mask is not None
        return grad_output * self.mask


# Weighted softmax loss for imbalanced classes.
class SoftmaxCrossEntropy:
    # Initializes layer parameters and caches.
    def __init__(self):
        self.probabilities: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.sample_weights: np.ndarray | None = None
        self.weight_sum: float = 1.0

    # Computes the forward pass.
    def forward(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        class_weights: dict[int, float] | None = None,
    ) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        sample_weights = np.ones(len(labels), dtype=np.float32)
        if class_weights is not None:
            sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float32)

        losses = -np.log(probabilities[np.arange(len(labels)), labels] + 1e-12)
        self.probabilities = probabilities
        self.labels = labels
        self.sample_weights = sample_weights
        self.weight_sum = float(np.sum(sample_weights) + 1e-12)
        return float(np.sum(losses * sample_weights) / self.weight_sum)

    # Computes gradients for backpropagation.
    def backward(self) -> np.ndarray:
        assert self.probabilities is not None and self.labels is not None and self.sample_weights is not None
        grad = self.probabilities.copy()
        grad[np.arange(len(self.labels)), self.labels] -= 1.0
        grad *= self.sample_weights[:, None] / self.weight_sum
        return grad


# Complete manual CNN model.
class CNNFromScratch:
    # Initializes layer parameters and caches.
    def __init__(self, input_length: int, num_classes: int, dropout_rate: float = 0.2):
        self.input_length = input_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.conv1 = Conv1D(in_channels=1, out_channels=16, kernel_size=7)
        self.relu1 = ReLU()
        self.pool1 = MaxPool1D(pool_size=2)

        self.conv2 = Conv1D(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = ReLU()
        self.pool2 = MaxPool1D(pool_size=2)

        self.conv3 = Conv1D(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = ReLU()
        self.global_pool = GlobalAveragePool1D()

        self.dense1 = Dense(64, 32)
        self.relu4 = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(32, num_classes)
        self.loss = SoftmaxCrossEntropy()

    # Computes the forward pass.
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.conv1.forward(x, training=training)
        x = self.relu1.forward(x, training=training)
        x = self.pool1.forward(x, training=training)

        x = self.conv2.forward(x, training=training)
        x = self.relu2.forward(x, training=training)
        x = self.pool2.forward(x, training=training)

        x = self.conv3.forward(x, training=training)
        x = self.relu3.forward(x, training=training)
        x = self.global_pool.forward(x, training=training)

        x = self.dense1.forward(x, training=training)
        x = self.relu4.forward(x, training=training)
        x = self.dropout.forward(x, training=training)
        return self.output_layer.forward(x, training=training)

    # Computes gradients for backpropagation.
    def backward(self, grad_output: np.ndarray) -> None:
        grad = self.output_layer.backward(grad_output)
        grad = self.dropout.backward(grad)
        grad = self.relu4.backward(grad)
        grad = self.dense1.backward(grad)
        grad = self.global_pool.backward(grad)

        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad)

    # Updates trainable weights.
    def update(self, learning_rate: float, optimizer: str = "sgd") -> None:
        self.conv1.update(learning_rate, optimizer)
        self.conv2.update(learning_rate, optimizer)
        self.conv3.update(learning_rate, optimizer)
        self.dense1.update(learning_rate, optimizer)
        self.output_layer.update(learning_rate, optimizer)

    # Predicts class ids from logits.
    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)

    # Returns all trainable weights.
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

    # Loads saved trainable weights.
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

    # Counts trainable parameters.
    def parameter_count(self) -> int:
        return int(sum(np.prod(value.shape) for value in self.get_parameters().values()))

    # Exports trained NumPy weights to Keras.
    def save_keras_model(self, path: str | Path) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required only for exporting the scratch CNN to Keras.") from exc

        inputs = tf.keras.Input(shape=(self.input_length, 1), name="ecg_input")
        x = tf.keras.layers.Conv1D(16, 7, padding="same", activation=None, name="conv1")(inputs)
        x = tf.keras.layers.ReLU(name="relu1")(x)
        x = tf.keras.layers.MaxPool1D(2, name="pool1")(x)
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation=None, name="conv2")(x)
        x = tf.keras.layers.ReLU(name="relu2")(x)
        x = tf.keras.layers.MaxPool1D(2, name="pool2")(x)
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation=None, name="conv3")(x)
        x = tf.keras.layers.ReLU(name="relu3")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool")(x)
        x = tf.keras.layers.Dense(32, activation=None, name="dense1")(x)
        x = tf.keras.layers.ReLU(name="relu4")(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="output_layer")(x)

        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="scratch_ecg_cnn")
        keras_model.get_layer("conv1").set_weights([self.conv1.weights.transpose(2, 1, 0), self.conv1.bias])
        keras_model.get_layer("conv2").set_weights([self.conv2.weights.transpose(2, 1, 0), self.conv2.bias])
        keras_model.get_layer("conv3").set_weights([self.conv3.weights.transpose(2, 1, 0), self.conv3.bias])
        keras_model.get_layer("dense1").set_weights([self.dense1.weights, self.dense1.bias])
        keras_model.get_layer("output_layer").set_weights([self.output_layer.weights, self.output_layer.bias])
        keras_model.save(path)


# Applies one Adam optimizer update.
def adam_update(
    params: np.ndarray,
    grads: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> np.ndarray:
    first_moment *= beta1
    first_moment += (1.0 - beta1) * grads
    second_moment *= beta2
    second_moment += (1.0 - beta2) * (grads * grads)
    first_unbiased = first_moment / (1.0 - beta1**step)
    second_unbiased = second_moment / (1.0 - beta2**step)
    return params - learning_rate * first_unbiased / (np.sqrt(second_unbiased) + eps)


# Builds the scratch CNN.
def build_tiny_cnn(input_length: int, num_classes: int, dropout_rate: float = 0.2) -> CNNFromScratch:
    return CNNFromScratch(input_length=input_length, num_classes=num_classes, dropout_rate=dropout_rate)


# Backward-compatible model builder.
def build_improved_cnn(input_length: int, num_classes: int, dropout_rate: float = 0.2) -> CNNFromScratch:
    return build_tiny_cnn(input_length=input_length, num_classes=num_classes, dropout_rate=dropout_rate)
