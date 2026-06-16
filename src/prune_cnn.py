from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d import build_tiny_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a trained scratch 1D CNN and evaluate structured compression effects."
    )
    parser.add_argument("--weights", type=Path, default=Path("results/baseline_cnn/tiny_ecg_cnn_weights.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_cnn/pruning"))
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--structured",
        action="store_true",
        default=True,
        help="Use structured filter/channel and hidden-neuron pruning. This is the default WP7 method.",
    )
    method_group.add_argument(
        "--magnitude",
        action="store_false",
        dest="structured",
        help="Use unstructured magnitude pruning as a secondary comparison.",
    )
    parser.add_argument(
        "--prune-fractions",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,0.9",
        help="Comma-separated fractions of filters/neurons to prune.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "standard", "per_sample"],
        default="none",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.weights.exists() and args.weights.suffix == ".npz":
        fallback = args.weights.with_suffix(".keras")
        if fallback.exists():
            print(f"[prune_cnn] Weight .npz file not found; falling back to Keras model {fallback}")
            args.weights = fallback

    dataset = (
        make_demo_dataset()
        if args.demo_data
        else load_mitbih_csv(
            data_dir=args.data_dir,
            normalize=args.normalize,
        )
    )

    model = build_tiny_cnn(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
    )
    weights = _load_weights(args.weights)
    model.set_parameters(weights)

    prune_fractions = _parse_prune_fractions(args.prune_fractions)
    pruning_method = "structured" if args.structured else "magnitude"
    records: list[dict[str, float | str]] = []

    for prune_fraction in prune_fractions:
        pruned_model = copy.deepcopy(model)
        if args.structured:
            pruned_model.prune_structured_by_fraction(prune_fraction)
        else:
            pruned_model.prune_magnitude_by_fraction(prune_fraction)

        val_loss, val_accuracy = _evaluate_model(pruned_model, dataset.x_val, dataset.y_val, args.batch_size)
        test_loss, test_accuracy = _evaluate_model(pruned_model, dataset.x_test, dataset.y_test, args.batch_size)
        sparsity = pruned_model.sparsity()
        total_params = pruned_model.total_parameters()
        zero_params = pruned_model.zero_parameters()
        nonzero_params = total_params - zero_params
        estimated_size_bytes = int(nonzero_params * 4)
        structured_counts = pruned_model.structured_counts()
        structured_sparsity = pruned_model.structured_sparsity()

        record = {
            "method": pruning_method,
            "prune_fraction": prune_fraction,
            "sparsity": sparsity,
            "structured_sparsity": structured_sparsity,
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "nonzero_parameters": nonzero_params,
            "estimated_size_bytes": estimated_size_bytes,
            **structured_counts,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        records.append(record)

        prune_path = args.output_dir / f"pruned_{int(prune_fraction * 100):02d}.npz"
        np.savez(prune_path, **pruned_model.get_parameters())
        print(
            f"[prune_cnn] method={pruning_method} fraction={prune_fraction:.2f} "
            f"parameter_sparsity={sparsity:.4f} structured_sparsity={structured_sparsity:.4f} "
            f"val_acc={val_accuracy:.4f} test_acc={test_accuracy:.4f} "
            f"size_est={estimated_size_bytes}"
        )

    results = pd.DataFrame(records)
    summary_path = args.output_dir / "pruning_metrics.csv"
    results.to_csv(summary_path, index=False)
    (args.output_dir / "pruning_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    baseline_metrics = _load_baseline_metrics(args.output_dir)
    plot_pruning_tradeoffs(results, args.output_dir, pruning_method)
    write_pruning_summary(results, args.output_dir, pruning_method, baseline_metrics)

    print(f"[prune_cnn] saved pruning summary to {summary_path}")


def _load_weights(weights_path: Path) -> dict[str, np.ndarray]:
    if weights_path.suffix == ".npz":
        if not weights_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        with np.load(weights_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    if weights_path.suffix in {".keras", ".h5", ".hdf5"}:
        if not weights_path.exists():
            raise FileNotFoundError(f"Keras model file not found: {weights_path}")
        return _load_weights_from_keras(weights_path)

    raise ValueError(f"Unsupported weights path extension: {weights_path.suffix}")


def _load_weights_from_keras(model_path: Path) -> dict[str, np.ndarray]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load Keras model weights.") from exc

    keras_model = tf.keras.models.load_model(str(model_path))

    def _find_weights(layer_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        for name in layer_names:
            try:
                return keras_model.get_layer(name).get_weights()
            except ValueError:
                continue
        raise ValueError(f"None of the expected layers were found: {layer_names}")

    weights = {}
    weights["conv1.weights"], weights["conv1.bias"] = _find_weights(["conv1", "conv1d"])
    weights["conv2.weights"], weights["conv2.bias"] = _find_weights(["conv2", "conv1d_1"])
    weights["conv3.weights"], weights["conv3.bias"] = _find_weights(["conv3", "conv1d_2"])
    weights["dense1.weights"], weights["dense1.bias"] = _find_weights(["dense1", "dense"])
    weights["output_layer.weights"], weights["output_layer.bias"] = _find_weights(["output_layer", "class_probs"])

    # Transpose Keras conv weights from (kernel, in_channels, out_channels) to (out_channels, in_channels, kernel)
    weights["conv1.weights"] = weights["conv1.weights"].transpose(2, 1, 0)
    weights["conv2.weights"] = weights["conv2.weights"].transpose(2, 1, 0)
    weights["conv3.weights"] = weights["conv3.weights"].transpose(2, 1, 0)
    return weights


def _load_baseline_metrics(output_dir: Path) -> dict[str, float] | None:
    baseline_metrics_path = output_dir.parent / "metrics.json"
    if baseline_metrics_path.exists():
        try:
            return json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


def _parse_prune_fractions(raw_fractions: str) -> list[float]:
    fractions = [float(x.strip()) for x in raw_fractions.split(",") if x.strip()]
    if not fractions:
        raise ValueError("At least one prune fraction must be provided.")
    invalid = [fraction for fraction in fractions if not 0.0 <= fraction <= 1.0]
    if invalid:
        raise ValueError(f"Prune fractions must be between 0.0 and 1.0. Invalid values: {invalid}")
    return fractions


def _evaluate_model(model, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[float, float]:
    total_loss = 0.0
    predictions = []
    for start in range(0, len(x), batch_size):
        x_batch = x[start : start + batch_size]
        y_batch = y[start : start + batch_size]
        logits = model.forward(x_batch)
        batch_loss = model.loss.forward(logits, y_batch)
        total_loss += float(batch_loss) * len(x_batch)
        predictions.append(np.argmax(logits, axis=1))

    all_predictions = np.concatenate(predictions)
    accuracy = float(np.mean(all_predictions == y))
    return float(total_loss / len(x)), accuracy


def plot_pruning_tradeoffs(results: pd.DataFrame, output_dir: Path, pruning_method: str) -> None:
    method_label = pruning_method.capitalize()
    plt.figure(figsize=(8, 5))
    plt.plot(results["sparsity"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.plot(results["sparsity"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.title(f"{method_label} CNN pruning: parameter sparsity vs accuracy")
    plt.xlabel("Parameter sparsity")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_sparsity_vs_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["prune_fraction"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.plot(results["prune_fraction"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title(f"{method_label} CNN pruning: prune fraction vs accuracy")
    plt.xlabel("Prune fraction")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_fraction_vs_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["sparsity"], results["estimated_size_bytes"], marker="o", label="Estimated compressed size")
    plt.title(f"{method_label} CNN pruning: parameter sparsity vs estimated compressed size")
    plt.xlabel("Parameter sparsity")
    plt.ylabel("Estimated size (bytes)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_sparsity_vs_estimated_size.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["estimated_size_bytes"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.plot(results["estimated_size_bytes"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title(f"{method_label} CNN pruning: estimated size vs accuracy")
    plt.xlabel("Estimated size (bytes)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_size_vs_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["structured_sparsity"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.plot(results["structured_sparsity"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title(f"{method_label} CNN pruning: structured sparsity vs accuracy")
    plt.xlabel("Structured filter/neuron sparsity")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pruning_structured_sparsity_vs_accuracy.png", dpi=180)
    plt.close()


def write_pruning_summary(
    results: pd.DataFrame,
    output_dir: Path,
    pruning_method: str,
    baseline_metrics: dict[str, float] | None = None,
) -> None:
    method_label = (
        "structured filter/channel and hidden-neuron pruning"
        if pruning_method == "structured"
        else "unstructured magnitude pruning"
    )
    lines = [
        "# CNN Pruning Evaluation Summary",
        "",
        f"This report documents {method_label} for the scratch 1D CNN.",
        "",
        "The WP7 target is structured pruning. Magnitude pruning is available only as a secondary comparison.",
        "",
        "Structured pruning deactivates whole convolution filters, propagates pruned channels into downstream convolution inputs, deactivates hidden dense neurons, and keeps output class logits intact.",
        "",
        "Pruning here stores dense arrays with zeros. It does not physically rebuild a smaller CNN architecture.",
        "",
        "## Summary metrics",
        "",
        f"- Pruning method: {pruning_method}",
        f"- Prune fractions: {', '.join(str(x) for x in results['prune_fraction'].tolist())}",
        f"- Parameter sparsity values: {', '.join(f'{x:.4f}' for x in results['sparsity'].tolist())}",
        f"- Structured sparsity values: {', '.join(f'{x:.4f}' for x in results['structured_sparsity'].tolist())}",
        "",
        "## Best observed results",
        "",
    ]

    best_val = results.loc[results["val_accuracy"].idxmax()]
    best_test = results.loc[results["test_accuracy"].idxmax()]

    lines.extend([
        f"- Best validation accuracy: {best_val['val_accuracy']:.4f} at sparsity {best_val['sparsity']:.4f}",
        f"- Best test accuracy: {best_test['test_accuracy']:.4f} at sparsity {best_test['sparsity']:.4f}",
        "",
        "## Plots",
        "",
        "- `pruning_sparsity_vs_accuracy.png`",
        "- `pruning_fraction_vs_accuracy.png`",
        "- `pruning_sparsity_vs_estimated_size.png`",
        "- `pruning_size_vs_accuracy.png`",
        "- `pruning_structured_sparsity_vs_accuracy.png`",
        "",
        "## Full results table",
        "",
        "| method | prune_fraction | parameter_sparsity | structured_sparsity | active_conv_filters | pruned_conv_filters | active_hidden_neurons | pruned_hidden_neurons | nonzero_parameters | estimated_size_bytes | val_accuracy | test_accuracy | val_loss | test_loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in results.iterrows():
        lines.append(
            f"| {row['method']} | {row['prune_fraction']:.2f} | {row['sparsity']:.4f} | "
            f"{row['structured_sparsity']:.4f} | {int(row['active_conv_filters'])} | "
            f"{int(row['pruned_conv_filters'])} | {int(row['active_hidden_neurons'])} | "
            f"{int(row['pruned_hidden_neurons'])} | {int(row['nonzero_parameters'])} | "
            f"{int(row['estimated_size_bytes'])} | {row['val_accuracy']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['val_loss']:.4f} | {row['test_loss']:.4f} |"
        )

    if baseline_metrics is not None:
        lines.extend([
            "",
            "## Original baseline model",
            "",
            f"- Baseline test accuracy: {baseline_metrics.get('test_accuracy', float('nan')):.4f}",
            f"- Baseline model size: {baseline_metrics.get('keras_model_size_bytes', baseline_metrics.get('model_size_bytes', 'unknown'))} bytes",
            "",
            "The pruning results below compare pruned models against this original baseline model.",
            "Estimated compressed size is derived from the number of nonzero float32 parameters, not from an actually smaller saved model file.",
            "",
        ])

    lines.extend([
        "",
        "## Size interpretation",
        "",
        "`estimated_size_bytes` assumes only nonzero float32 parameters are stored. The `.npz` checkpoint still contains dense arrays with zeros, so actual file size reduction requires sparse storage, architecture shrinking, or a deployment-specific compression format.",
        "",
    ])

    output_path = output_dir / "pruning_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

