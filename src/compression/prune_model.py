from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d import build_baseline_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a baseline ECG model using structured filter/block pruning.")
    parser.add_argument("--model", type=Path, default=Path("results/baseline_cnn/baseline_ecg_cnn.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pruned"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--final-sparsity", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true", help="Use synthetic ECG-like data instead of real CSV data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    model = _load_or_build_model(args.model, dataset.input_length, dataset.num_classes)
    pruned_model = _build_pruned_model(
        model,
        args.final_sparsity,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        len(dataset.x_train),
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    history = pruned_model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    stripped_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model_path = args.output_dir / "baseline_ecg_cnn_pruned.keras"
    stripped_model.save(model_path)

    test_loss, test_accuracy = stripped_model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    probabilities = stripped_model.predict(dataset.x_test, batch_size=args.batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)

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
        "parameters": int(stripped_model.count_params()),
        "final_sparsity": args.final_sparsity,
        "model_path": str(model_path),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    print(json.dumps(metrics, indent=2))


def _load_or_build_model(model_path: Path, input_length: int, num_classes: int) -> tf.keras.Model:
    if model_path.exists():
        import keras

        keras_model = keras.models.load_model(model_path, compile=False)
        baseline_model = build_baseline_cnn(input_length=input_length, num_classes=num_classes)

        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            temp_weights_path = Path(tmp.name)

        try:
            keras_model.save_weights(temp_weights_path)
            baseline_model.load_weights(temp_weights_path)
        finally:
            if temp_weights_path.exists():
                temp_weights_path.unlink()

        return baseline_model

    return build_baseline_cnn(input_length=input_length, num_classes=num_classes)


def _build_pruned_model(
    model: tf.keras.Model,
    final_sparsity: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    train_size: int,
) -> tf.keras.Model:
    import keras

    train_steps = int(np.ceil(train_size / batch_size)) if epochs > 0 else 1
    end_step = train_steps * epochs
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=end_step,
    )

    def prune_layer(layer: keras.layers.Layer) -> keras.layers.Layer:
        if isinstance(layer, keras.layers.Conv1D):
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer,
                pruning_schedule=pruning_schedule,
            )
        if isinstance(layer, keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer,
                pruning_schedule=pruning_schedule,
                block_size=(1, layer.units),
                block_pooling_type="AVG",
            )
        return layer

    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=prune_layer,
        input_tensors=model.inputs,
    )
    pruned_model.set_weights(model.get_weights())
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return pruned_model


def _save_training_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(history_df["accuracy"], label="train accuracy")
    plt.plot(history_df["val_accuracy"], label="val accuracy")
    plt.plot(history_df["loss"], label="train loss")
    plt.plot(history_df["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Pruning Training History")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
