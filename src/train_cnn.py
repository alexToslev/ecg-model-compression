from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.data.visualization import visualize_dataset
from src.evaluation.summarize_baseline import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    write_summary,
)
from src.models.cnn1d import build_improved_cnn


# Defines command-line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch 1D CNN with class weighting and rare-class augmentation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/improved_cnn_scratch"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--class-weight-cap", type=float, default=10.0)
    parser.add_argument("--augment-rare-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rare-target-count", type=int, default=2000)
    parser.add_argument("--rare-threshold", type=float, default=0.2)
    return parser.parse_args()


# Runs the full training workflow.
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args)
    save_dataset_visualizations(dataset, args)

    class_weights = (
        calculate_class_weights(dataset.y_train, dataset.num_classes, args.class_weight_cap)
        if args.class_weights == "balanced"
        else None
    )
    x_train, y_train, augmentation_report = prepare_training_data(dataset, args)
    if args.max_train_samples is not None and args.max_train_samples < len(x_train):
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(x_train), size=args.max_train_samples, replace=False)
        x_train = x_train[chosen]
        y_train = y_train[chosen]

    model = build_improved_cnn(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
        dropout_rate=args.dropout_rate,
    )

    print_dataset_summary(dataset, x_train, args, class_weights, augmentation_report)
    start_time = time.perf_counter()
    history = train_model(model, x_train, y_train, dataset, args, class_weights)
    training_seconds = time.perf_counter() - start_time

    save_artifacts(model, history, dataset, args, class_weights, augmentation_report, training_seconds)


# Loads demo or real MIT-BIH data.
def load_dataset(args: argparse.Namespace):
    if args.demo_data:
        return make_demo_dataset(validation_fraction=args.validation_fraction, seed=args.seed)
    return load_mitbih_csv(
        data_dir=args.data_dir,
        validation_fraction=args.validation_fraction,
        normalize=args.normalize,
        seed=args.seed,
    )


# Trains the CNN batch by batch.
def train_model(model, x_train, y_train, dataset, args, class_weights):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(x_train))
        x_epoch = x_train[order]
        y_epoch = y_train[order]
        epoch_loss = 0.0

        for start in range(0, len(x_epoch), args.batch_size):
            x_batch = x_epoch[start : start + args.batch_size]
            y_batch = y_epoch[start : start + args.batch_size]

            logits = model.forward(x_batch, training=True)
            batch_loss = model.loss.forward(logits, y_batch, class_weights=class_weights)
            grad_logits = model.loss.backward()
            model.backward(grad_logits)
            model.update(args.learning_rate, optimizer=args.optimizer)
            epoch_loss += batch_loss * len(x_batch)

        train_loss = epoch_loss / len(x_train)
        train_accuracy = accuracy(model, x_train, y_train, args.batch_size)
        val_loss, val_accuracy = evaluate_loss_accuracy(model, dataset.x_val, dataset.y_val, args.batch_size, class_weights)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        print(
            f"epoch={epoch} loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

    return history


# Computes loss and accuracy on a split.
def evaluate_loss_accuracy(model, x, y, batch_size: int, class_weights=None) -> tuple[float, float]:
    total_loss = 0.0
    predictions = []
    for start in range(0, len(x), batch_size):
        x_batch = x[start : start + batch_size]
        y_batch = y[start : start + batch_size]
        logits = model.forward(x_batch, training=False)
        total_loss += model.loss.forward(logits, y_batch, class_weights=class_weights) * len(x_batch)
        predictions.append(np.argmax(logits, axis=1))
    y_pred = np.concatenate(predictions)
    return float(total_loss / len(x)), float(np.mean(y_pred == y))


# Computes prediction accuracy.
def accuracy(model, x, y, batch_size: int) -> float:
    predictions = predict_in_batches(model, x, batch_size)
    return float(np.mean(predictions == y))


# Predicts labels in smaller batches.
def predict_in_batches(model, x, batch_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(x), batch_size):
        logits = model.forward(x[start : start + batch_size], training=False)
        predictions.append(np.argmax(logits, axis=1))
    return np.concatenate(predictions)


# Computes capped class-imbalance penalties.
def calculate_class_weights(labels: np.ndarray, num_classes: int, weight_cap: float) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    weights = total / (num_classes * np.maximum(counts, 1.0))
    weights = np.minimum(weights, weight_cap)
    return {class_id: float(weights[class_id]) for class_id in range(num_classes)}


# Applies optional rare-class augmentation.
def prepare_training_data(dataset, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    if not args.augment_rare_classes:
        return dataset.x_train, dataset.y_train, {"enabled": False}

    return augment_rare_classes(
        dataset.x_train,
        dataset.y_train,
        num_classes=dataset.num_classes,
        target_count=args.rare_target_count,
        rare_threshold=args.rare_threshold,
        seed=args.seed,
    )


# Creates extra samples for rare classes.
def augment_rare_classes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    target_count: int,
    rare_threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=num_classes)
    max_count = int(np.max(counts))
    rare_limit = max(1, int(max_count * rare_threshold))

    augmented_x = [x_train]
    augmented_y = [y_train]
    generated_per_class = {}

    for class_id, count in enumerate(counts):
        if count == 0 or count >= rare_limit:
            generated_per_class[class_id] = 0
            continue

        desired_count = min(target_count, max_count)
        needed = max(0, desired_count - int(count))
        class_samples = x_train[y_train == class_id]
        new_samples = np.empty((needed, *x_train.shape[1:]), dtype=np.float32)

        for index in range(needed):
            base = class_samples[rng.integers(0, len(class_samples))]
            new_samples[index] = augment_one_sample(base, rng)

        augmented_x.append(new_samples)
        augmented_y.append(np.full(needed, class_id, dtype=y_train.dtype))
        generated_per_class[class_id] = int(needed)

    x_out = np.concatenate(augmented_x, axis=0)
    y_out = np.concatenate(augmented_y, axis=0)
    order = rng.permutation(len(y_out))

    return (
        x_out[order],
        y_out[order],
        {
            "enabled": True,
            "rare_threshold": rare_threshold,
            "rare_limit": rare_limit,
            "target_count": target_count,
            "original_counts": {str(i): int(counts[i]) for i in range(num_classes)},
            "generated_per_class": {str(k): int(v) for k, v in generated_per_class.items()},
            "final_counts": {str(i): int(np.sum(y_out == i)) for i in range(num_classes)},
        },
    )


# Applies mild ECG-safe augmentation.
def augment_one_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = sample.astype(np.float32).copy()
    augmented *= rng.uniform(0.95, 1.05)
    augmented += rng.normal(0.0, 0.005, size=augmented.shape).astype(np.float32)
    augmented += rng.uniform(-0.01, 0.01)

    shift = int(rng.integers(-2, 3))
    if shift != 0:
        augmented = np.roll(augmented, shift=shift, axis=0)
        if shift > 0:
            augmented[:shift] = augmented[shift]
        else:
            augmented[shift:] = augmented[shift - 1]
    return augmented.astype(np.float32)


# Saves metrics, plots, weights, and reports.
def save_artifacts(model, history, dataset, args, class_weights, augmentation_report, training_seconds: float) -> None:
    np.savez(args.output_dir / "tiny_ecg_cnn_weights.npz", **model.get_parameters())

    keras_model_path = args.output_dir / "tiny_ecg_cnn.keras"
    keras_model_saved = False
    try:
        model.save_keras_model(keras_model_path)
        keras_model_saved = True
    except RuntimeError as exc:
        print(f"Warning: Keras export skipped: {exc}")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output_dir / "history.csv", index=False)

    test_loss, test_accuracy = evaluate_loss_accuracy(
        model, dataset.x_test, dataset.y_test, args.batch_size, class_weights
    )
    predictions = predict_in_batches(model, dataset.x_test, args.batch_size)
    report = classification_report(dataset.y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(dataset.y_test, predictions)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "num_classes": dataset.num_classes,
        "input_length": dataset.input_length,
        "architecture": "from_scratch_weighted_augmented_1d_cnn",
        "layers": [
            "Conv1D(16,k=7)",
            "ReLU",
            "MaxPool1D",
            "Conv1D(32,k=5)",
            "ReLU",
            "MaxPool1D",
            "Conv1D(64,k=3)",
            "ReLU",
            "GlobalAveragePool1D",
            "Dense(32)",
            "ReLU",
            "Dropout",
            "Dense(5)",
            "WeightedSoftmaxCrossEntropy",
        ],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "dropout_rate": args.dropout_rate,
        "validation_fraction": args.validation_fraction,
        "normalize": args.normalize,
        "seed": args.seed,
        "class_weights": None if class_weights is None else {str(k): v for k, v in class_weights.items()},
        "class_weight_cap": args.class_weight_cap,
        "augmentation": augmentation_report,
        "parameters": model.parameter_count(),
        "training_seconds": float(training_seconds),
        "weights_path": str(args.output_dir / "tiny_ecg_cnn_weights.npz"),
    }
    if keras_model_saved:
        metrics["keras_model_path"] = str(keras_model_path)
        metrics["keras_model_size_bytes"] = int(keras_model_path.stat().st_size)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(history_frame, plots_dir / "learning_curves.png")
    plot_class_metrics(report, plots_dir / "class_metrics.png")
    plot_confusion_matrix(matrix, plots_dir / "confusion_matrix.png")
    write_summary(
        args.output_dir / "baseline_summary.md",
        history_frame,
        metrics,
        report,
        matrix,
        title="From-Scratch Improved CNN Summary",
        model_description=(
            "A manual NumPy CNN was trained with explicit forward pass, backward pass, "
            "weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation."
        ),
        next_step="Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.",
    )
    print(json.dumps(metrics, indent=2))


# Saves dataset diagnostic plots.
def save_dataset_visualizations(dataset, args: argparse.Namespace) -> None:
    visualize_dataset(dataset, args.output_dir / "dataset_visualizations")


# Prints the experiment setup.
def print_dataset_summary(dataset, x_train, args, class_weights, augmentation_report) -> None:
    print("[train_cnn] From-scratch CNN training")
    print(f"  original train samples: {len(dataset.x_train)}")
    print(f"  effective train samples: {len(x_train)}")
    print(f"  validation samples: {len(dataset.x_val)}")
    print(f"  test samples: {len(dataset.x_test)}")
    print(f"  input length: {dataset.input_length}")
    print(f"  num classes: {dataset.num_classes}")
    print(f"  optimizer: {args.optimizer}")
    print(f"  class weights: {class_weights}")
    print(f"  augmentation: {json.dumps(augmentation_report)}")


if __name__ == "__main__":
    main()
