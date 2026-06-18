from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an improved 1D CNN for imbalanced MIT-BIH heartbeat classification."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/improved_cnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument(
        "--class-weights",
        choices=["balanced", "none"],
        default="balanced",
        help="Use a stronger loss penalty for rare classes.",
    )
    parser.add_argument(
        "--augment-rare-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create mild augmented copies for rare training classes.",
    )
    parser.add_argument(
        "--rare-target-count",
        type=int,
        default=2000,
        help="Maximum target count for each rare class after augmentation.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.2,
        help="Classes below this fraction of the largest class are treated as rare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args)
    save_dataset_visualizations(dataset, args)

    class_weights = (
        calculate_class_weights(dataset.y_train, dataset.num_classes)
        if args.class_weights == "balanced"
        else None
    )
    x_train, y_train, augmentation_report = prepare_training_data(dataset, args)

    model = build_improved_cnn(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
        dropout_rate=args.dropout_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    print_dataset_summary(dataset, args, class_weights, augmentation_report)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    save_artifacts(model, history, dataset, args, class_weights, augmentation_report)


def load_dataset(args: argparse.Namespace):
    if args.demo_data:
        print("[train_cnn] Using synthetic demo dataset.")
        return make_demo_dataset(
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

    print(f"[train_cnn] Loading MIT-BIH dataset from {args.data_dir}")
    return load_mitbih_csv(
        data_dir=args.data_dir,
        validation_fraction=args.validation_fraction,
        normalize=args.normalize,
        seed=args.seed,
    )


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return {class_id: float(weights[class_id]) for class_id in range(num_classes)}


def prepare_training_data(dataset, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    if not args.augment_rare_classes:
        return dataset.x_train, dataset.y_train, {"enabled": False}

    x_aug, y_aug, report = augment_rare_classes(
        dataset.x_train,
        dataset.y_train,
        num_classes=dataset.num_classes,
        target_count=args.rare_target_count,
        rare_threshold=args.rare_threshold,
        seed=args.seed,
    )
    return x_aug, y_aug, report


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
    generated_per_class: dict[int, int] = {}

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
            "final_counts": {
                str(i): int(np.sum(y_out == i))
                for i in range(num_classes)
            },
        },
    )


def augment_one_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = sample.astype(np.float32).copy()
    augmented *= rng.uniform(0.9, 1.1)
    augmented += rng.normal(0.0, 0.01, size=augmented.shape).astype(np.float32)
    augmented += rng.uniform(-0.02, 0.02)

    shift = int(rng.integers(-3, 4))
    if shift != 0:
        augmented = np.roll(augmented, shift=shift, axis=0)
        if shift > 0:
            augmented[:shift] = augmented[shift]
        else:
            augmented[shift:] = augmented[shift - 1]

    return augmented.astype(np.float32)


def save_artifacts(
    model: tf.keras.Model,
    history,
    dataset,
    args: argparse.Namespace,
    class_weights: dict[int, float] | None,
    augmentation_report: dict,
) -> None:
    model_path = args.output_dir / "tiny_ecg_cnn.keras"
    model.save(model_path)

    history_frame = pd.DataFrame(
        {
            "train_loss": history.history["loss"],
            "train_accuracy": history.history["accuracy"],
            "val_loss": history.history["val_loss"],
            "val_accuracy": history.history["val_accuracy"],
        }
    )
    history_frame.to_csv(args.output_dir / "history.csv", index=False)

    test_loss, test_accuracy = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    probabilities = model.predict(dataset.x_test, batch_size=args.batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(dataset.y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(dataset.y_test, predictions)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "num_classes": dataset.num_classes,
        "input_length": dataset.input_length,
        "architecture": "improved_keras_1d_cnn",
        "layers": [
            "Conv1D(16,k=7)",
            "BatchNorm",
            "ReLU",
            "MaxPool1D",
            "Conv1D(32,k=5)",
            "BatchNorm",
            "ReLU",
            "MaxPool1D",
            "Conv1D(64,k=3)",
            "BatchNorm",
            "ReLU",
            "GlobalAveragePooling1D",
            "Dense(32)",
            "Dropout",
            "Dense(5,softmax)",
        ],
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_frame),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "dropout_rate": args.dropout_rate,
        "validation_fraction": args.validation_fraction,
        "normalize": args.normalize,
        "seed": args.seed,
        "class_weights": None if class_weights is None else {str(k): v for k, v in class_weights.items()},
        "augmentation": augmentation_report,
        "parameters": int(model.count_params()),
        "keras_model_path": str(model_path),
        "keras_model_size_bytes": int(model_path.stat().st_size),
    }

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
        title="Improved 1D CNN Summary",
        model_description=(
            "An improved Keras 1D CNN was trained with class-weighted loss, Adam, "
            "BatchNorm, GlobalAveragePooling, Dropout, and optional rare-class augmentation."
        ),
        next_step=(
            "Compare macro F1 and minority-class recall against the previous CNN/MLP results, "
            "then quantize the best model for ESP32 testing."
        ),
    )

    print(json.dumps(metrics, indent=2))


def save_dataset_visualizations(dataset, args: argparse.Namespace) -> None:
    visualization_dir = args.output_dir / "dataset_visualizations"
    visualize_dataset(dataset, visualization_dir)


def print_dataset_summary(dataset, args: argparse.Namespace, class_weights, augmentation_report: dict) -> None:
    print("[train_cnn] Dataset summary:")
    print(f"  train samples: {len(dataset.x_train)}")
    print(f"  validation samples: {len(dataset.x_val)}")
    print(f"  test samples: {len(dataset.x_test)}")
    print(f"  input length: {dataset.input_length}")
    print(f"  num classes: {dataset.num_classes}")
    print(f"  normalization: {args.normalize}")
    print(f"  optimizer: Adam")
    print(f"  class weights: {class_weights}")
    print(f"  augmentation: {json.dumps(augmentation_report)}")


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


if __name__ == "__main__":
    main()
