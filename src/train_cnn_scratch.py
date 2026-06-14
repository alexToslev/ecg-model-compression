from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d_scratch import ScratchCNN1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D CNN from scratch with NumPy backpropagation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scratch_cnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic ECG-like data for a quick smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_data:
        dataset = make_demo_dataset(validation_fraction=args.validation_fraction, seed=args.seed)
    else:
        dataset = load_mitbih_csv(
            data_dir=args.data_dir,
            validation_fraction=args.validation_fraction,
            normalize=args.normalize,
            seed=args.seed,
        )

    x_train, y_train = limit_training_set(dataset.x_train, dataset.y_train, args.max_train_samples, rng)
    model = ScratchCNN1D(input_length=dataset.input_length, num_classes=dataset.num_classes, seed=args.seed)

    history_rows = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(model, x_train, y_train, args.batch_size, args.learning_rate, rng)
        train_accuracy = accuracy(model, x_train, y_train)
        val_accuracy = accuracy(model, dataset.x_val, dataset.y_val)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        history_rows.append(row)
        print(
            f"epoch={epoch} "
            f"loss={row['loss']:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    training_seconds = time.perf_counter() - start_time
    predictions = model.predict(dataset.x_test)
    test_accuracy = float((predictions == dataset.y_test).mean())
    report = classification_report(dataset.y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(dataset.y_test, predictions)

    pd.DataFrame(history_rows).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    metrics = {
        "test_accuracy": test_accuracy,
        "input_length": dataset.input_length,
        "num_classes": dataset.num_classes,
        "parameters": model.count_params(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "training_seconds": training_seconds,
        "implementation": "NumPy manual Conv1D/ReLU/MaxPool/Dense/SoftmaxCE/SGD",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "scratch_summary.md", metrics)
    print(json.dumps(metrics, indent=2))


def train_one_epoch(
    model: ScratchCNN1D,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    learning_rate: float,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.permutation(len(x_train))
    losses = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        loss = model.train_batch(x_train[batch_indices], y_train[batch_indices], learning_rate)
        losses.append(loss)
    return losses


def accuracy(model: ScratchCNN1D, x: np.ndarray, y: np.ndarray) -> float:
    predictions = model.predict(x)
    return float((predictions == y).mean())


def limit_training_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_train_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_train_samples <= 0 or len(x_train) <= max_train_samples:
        return x_train, y_train
    indices = rng.choice(len(x_train), size=max_train_samples, replace=False)
    return x_train[indices], y_train[indices]


def write_summary(output_path: Path, metrics: dict) -> None:
    lines = [
        "# From-Scratch CNN Summary",
        "",
        "This run trains a small 1D CNN with manual NumPy code for the learning parts.",
        "",
        "Implemented from scratch:",
        "",
        "- Conv1D forward and backward pass",
        "- ReLU forward and backward pass",
        "- MaxPool1D forward and backward pass",
        "- Global average pooling backward pass",
        "- Dense layer forward and backward pass",
        "- Softmax cross-entropy loss and gradient",
        "- Mini-batch SGD weight updates",
        "",
        "This branch is for learning and comparison. The TensorFlow/Keras branch remains the practical path for TensorFlow Lite and ESP32 deployment.",
        "",
        "## Results",
        "",
        f"- Test accuracy: {metrics['test_accuracy']:.4f}",
        f"- Parameters: {metrics['parameters']}",
        f"- Training seconds: {metrics['training_seconds']:.2f}",
        f"- Max train samples: {metrics['max_train_samples']}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
