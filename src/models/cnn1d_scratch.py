from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Conv1D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rng: np.random.Generator) -> None:
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        scale = np.sqrt(2.0 / (kernel_size * in_channels))
        self.weights = rng.normal(0.0, scale, size=(kernel_size, in_channels, out_channels)).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.x: np.ndarray | None = None
        self.x_padded: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)), mode="constant")
        batch_size, input_length, _ = x.shape
        out_channels = self.weights.shape[2]
        output = np.zeros((batch_size, input_length, out_channels), dtype=np.float32)

        for position in range(input_length):
            window = self.x_padded[:, position : position + self.kernel_size, :]
            output[:, position, :] = np.tensordot(window, self.weights, axes=([1, 2], [0, 1])) + self.bias
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None or self.x_padded is None:
            raise RuntimeError("forward must be called before backward")

        self.grad_weights.fill(0.0)
        self.grad_bias = grad_output.sum(axis=(0, 1))
        grad_x_padded = np.zeros_like(self.x_padded)
        input_length = self.x.shape[1]

        for position in range(input_length):
            window = self.x_padded[:, position : position + self.kernel_size, :]
            self.grad_weights += np.tensordot(window, grad_output[:, position, :], axes=([0], [0]))
            grad_x_padded[:, position : position + self.kernel_size, :] += np.tensordot(
                grad_output[:, position, :],
                self.weights,
                axes=([1], [2]),
            )

        if self.padding == 0:
            return grad_x_padded
        return grad_x_padded[:, self.padding : -self.padding, :]

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


class ReLU:
    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self.mask

    def step(self, learning_rate: float) -> None:
        return None


class MaxPool1D:
    def __init__(self, pool_size: int = 2) -> None:
        self.pool_size = pool_size
        self.x: np.ndarray | None = None
        self.output_length: int | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        batch_size, input_length, channels = x.shape
        self.output_length = input_length // self.pool_size
        trimmed = x[:, : self.output_length * self.pool_size, :]
        windows = trimmed.reshape(batch_size, self.output_length, self.pool_size, channels)
        return windows.max(axis=2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None or self.output_length is None:
            raise RuntimeError("forward must be called before backward")

        batch_size, input_length, channels = self.x.shape
        grad_x = np.zeros_like(self.x)
        trimmed = self.x[:, : self.output_length * self.pool_size, :]
        windows = trimmed.reshape(batch_size, self.output_length, self.pool_size, channels)
        max_values = windows.max(axis=2, keepdims=True)
        mask = windows == max_values
        mask_count = mask.sum(axis=2, keepdims=True)
        grad_windows = mask * (grad_output[:, :, np.newaxis, :] / mask_count)
        grad_x[:, : self.output_length * self.pool_size, :] = grad_windows.reshape(
            batch_size,
            self.output_length * self.pool_size,
            channels,
        )
        return grad_x

    def step(self, learning_rate: float) -> None:
        return None


class GlobalAveragePool1D:
    def __init__(self) -> None:
        self.input_length: int | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_length = x.shape[1]
        return x.mean(axis=1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.input_length is None:
            raise RuntimeError("forward must be called before backward")
        return np.repeat(grad_output[:, np.newaxis, :] / self.input_length, self.input_length, axis=1)

    def step(self, learning_rate: float) -> None:
        return None


class Dense:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.weights = rng.normal(0.0, scale, size=(in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.x: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("forward must be called before backward")
        self.grad_weights = self.x.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weights.T

    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


@dataclass
class LossResult:
    loss: float
    grad_logits: np.ndarray


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> LossResult:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch_size = labels.shape[0]
    loss = -np.log(probabilities[np.arange(batch_size), labels] + 1e-12).mean()

    grad_logits = probabilities
    grad_logits[np.arange(batch_size), labels] -= 1.0
    grad_logits /= batch_size
    return LossResult(loss=float(loss), grad_logits=grad_logits.astype(np.float32))


class ScratchCNN1D:
    def __init__(self, input_length: int, num_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.layers = [
            Conv1D(in_channels=1, out_channels=4, kernel_size=5, rng=rng),
            ReLU(),
            MaxPool1D(pool_size=2),
            Conv1D(in_channels=4, out_channels=8, kernel_size=3, rng=rng),
            ReLU(),
            GlobalAveragePool1D(),
            Dense(in_features=8, out_features=num_classes, rng=rng),
        ]
        self.input_length = input_length
        self.num_classes = num_classes

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        logits = self.forward(x)
        loss_result = softmax_cross_entropy(logits, y)
        grad = loss_result.grad_logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        for layer in self.layers:
            layer.step(learning_rate)
        return loss_result.loss

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        predictions = []
        for start in range(0, len(x), batch_size):
            logits = self.forward(x[start : start + batch_size])
            predictions.append(logits.argmax(axis=1))
        return np.concatenate(predictions)

    def count_params(self) -> int:
        total = 0
        for layer in self.layers:
            if hasattr(layer, "weights"):
                total += int(layer.weights.size)
            if hasattr(layer, "bias"):
                total += int(layer.bias.size)
        return total
