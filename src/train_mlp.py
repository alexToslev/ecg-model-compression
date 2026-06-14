from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.mlp import build_manual_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline MLP for ECG heartbeat classification."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_mlp"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--normalize",
        choices=["none", "standard", "per_sample"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dense-layers", type=int, default=2)
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Use synthetic ECG-like data for fast development and testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    _log_dataset_summary(dataset, args)

    model = build_manual_mlp(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
        hidden_units=args.hidden_units,
        dense_layers=args.dense_layers,
    )

    history = train_model(model, dataset, args)
    _save_training_artifacts(model, history, dataset, args)


def _load_dataset(args: argparse.Namespace):
    if args.demo_data:
        print("[train_mlp] Using synthetic demo dataset for development.")
        return make_demo_dataset(
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

    print(f"[train_mlp] Loading real MIT-BIH dataset from {args.data_dir}")
    return load_mitbih_csv(
        data_dir=args.data_dir,
        validation_fraction=args.validation_fraction,
        normalize=args.normalize,
        seed=args.seed,
    )


def _log_dataset_summary(dataset, args: argparse.Namespace) -> None:
    print("[train_mlp] Dataset summary:")
    print(f"  train samples: {len(dataset.x_train)}")
    print(f"  validation samples: {len(dataset.x_val)}")
    print(f"  test samples: {len(dataset.x_test)}")
    print(f"  input length: {dataset.input_length}")
    print(f"  num classes: {dataset.num_classes}")
    print(f"  normalization: {args.normalize}")
    print(f"  seed: {args.seed}")
    print(f"  hidden units: {args.hidden_units}")
    print(f"  dense layers: {args.dense_layers}")


def train_model(model, dataset, args: argparse.Namespace) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    x_train = dataset.x_train
    y_train = dataset.y_train
    num_samples = len(x_train)

    for epoch in range(1, args.epochs + 1):
        permutation = np.random.permutation(num_samples)
        x_shuffled = x_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0.0
        for start in range(0, num_samples, args.batch_size):
            x_batch = x_shuffled[start : start + args.batch_size]
            y_batch = y_shuffled[start : start + args.batch_size]

            logits = model.forward(x_batch)
            batch_loss = model.loss.forward(logits, y_batch)
            grad_logits = model.loss.backward()
            model.backward(grad_logits)
            model.update(args.learning_rate)

            epoch_loss += float(batch_loss) * len(x_batch)

        train_loss = epoch_loss / num_samples
        train_accuracy = _calculate_accuracy(model, dataset.x_train, dataset.y_train)
        val_loss, val_accuracy = _evaluate_model(model, dataset.x_val, dataset.y_val)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"[train_mlp] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    return history


def _evaluate_model(model, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(x)
    loss = model.loss.forward(logits, y)
    predictions = model.predict(x)
    accuracy = float(np.mean(predictions == y))
    return float(loss), accuracy


def _calculate_accuracy(model, x: np.ndarray, y: np.ndarray) -> float:
    predictions = model.predict(x)
    return float(np.mean(predictions == y))


def _save_training_artifacts(
    model,
    history: dict[str, list[float]],
    dataset,
    args: argparse.Namespace,
) -> None:
    model_path = args.output_dir / "baseline_mlp_weights.npz"
    np.savez(model_path, **model.get_parameters())

    keras_model_path = args.output_dir / "baseline_mlp.keras"
    try:
        model.save_keras_model(keras_model_path)
        model_path_for_metrics = keras_model_path
    except RuntimeError:
        model_path_for_metrics = model_path

    history_path = args.output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    test_loss, test_accuracy = _evaluate_model(model, dataset.x_test, dataset.y_test)
    predictions = model.predict(dataset.x_test)
    report = classification_report(dataset.y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(dataset.y_test, predictions)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_classes": dataset.num_classes,
        "input_length": dataset.input_length,
        "parameters": sum(int(np.prod(value.shape)) for value in model.get_parameters().values()),
        "weights_path": str(model_path),
        "keras_model_path": str(model_path_for_metrics),
        "history_path": str(history_path),
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    print(json.dumps(metrics, indent=2))


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)


if __name__ == "__main__":
    main()
