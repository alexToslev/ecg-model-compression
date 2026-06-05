from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d import build_baseline_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny 1D CNN for ECG heartbeat classification.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic ECG-like data for a quick smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo_data:
        dataset = make_demo_dataset(
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        dataset = load_mitbih_csv(
            data_dir=args.data_dir,
            validation_fraction=args.validation_fraction,
            normalize=args.normalize,
            seed=args.seed,
        )

    model = build_baseline_cnn(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    probabilities = model.predict(dataset.x_test, batch_size=args.batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)

    model_path = args.output_dir / "baseline_ecg_cnn.keras"
    model.save(model_path)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(args.output_dir / "history.csv", index=False)
    _save_training_plot(history_df, args.output_dir / "training_history.png")

    report = classification_report(dataset.y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(dataset.y_test, predictions)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_classes": dataset.num_classes,
        "input_length": dataset.input_length,
        "parameters": int(model.count_params()),
        "model_path": str(model_path),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    print(json.dumps(metrics, indent=2))


def _save_training_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history_df["accuracy"], label="train accuracy")
    plt.plot(history_df["val_accuracy"], label="val accuracy")
    plt.plot(history_df["loss"], label="train loss")
    plt.plot(history_df["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Training History")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    main()
