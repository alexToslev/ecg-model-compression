from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.mlp import build_manual_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a trained manual MLP and evaluate structured sparsity vs accuracy."
    )
    parser.add_argument("--weights", type=Path, default=Path("results/baseline_mlp/baseline_mlp_weights.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_mlp/pruning_structured"))
    parser.add_argument("--demo-data", action="store_true")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--structured",
        action="store_true",
        default=True,
        help="Use structured hidden-neuron pruning. This is the default WP4 method.",
    )
    method_group.add_argument(
        "--magnitude",
        action="store_false",
        dest="structured",
        help="Use the older unstructured magnitude pruning path.",
    )
    parser.add_argument(
        "--prune-fractions",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,0.9",
        help="Comma-separated pruning fractions to evaluate (0.0-1.0).",
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

    dataset = (
        make_demo_dataset()
        if args.demo_data
        else load_mitbih_csv(
            data_dir=args.data_dir,
            normalize=args.normalize,
        )
    )

    model = build_manual_mlp(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
    )
    parameters = _load_weights(args.weights)
    model.set_parameters(parameters)

    prune_fractions = _parse_prune_fractions(args.prune_fractions)
    pruning_method = "structured" if args.structured else "magnitude"
    records: list[dict[str, float | str]] = []

    for prune_fraction in prune_fractions:
        pruned_model = copy.deepcopy(model)
        if args.structured:
            pruned_model.prune_structured_by_fraction(prune_fraction)
        else:
            pruned_model.prune_by_fraction(prune_fraction)

        val_loss, val_accuracy = _evaluate_model(pruned_model, dataset.x_val, dataset.y_val)
        test_loss, test_accuracy = _evaluate_model(pruned_model, dataset.x_test, dataset.y_test)
        sparsity = pruned_model.sparsity()
        structured_counts = pruned_model.structured_neuron_counts()
        structured_sparsity = pruned_model.structured_sparsity()
        total_parameters = pruned_model.total_parameters()
        zero_parameters = int(round(sparsity * total_parameters))
        nonzero_parameters = total_parameters - zero_parameters
        estimated_size_bytes = int(nonzero_parameters * 4)

        record = {
            "method": pruning_method,
            "prune_fraction": prune_fraction,
            "sparsity": sparsity,
            "structured_sparsity": structured_sparsity,
            "total_parameters": total_parameters,
            "zero_parameters": zero_parameters,
            "nonzero_parameters": nonzero_parameters,
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
            f"[prune_mlp] method={pruning_method} fraction={prune_fraction:.2f} "
            f"sparsity={sparsity:.4f} structured_sparsity={structured_sparsity:.4f} "
            f"val_acc={val_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )

    results = pd.DataFrame(records)
    summary_path = args.output_dir / "pruning_metrics.csv"
    results.to_csv(summary_path, index=False)
    (args.output_dir / "pruning_metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    plot_pruning_tradeoffs(results, args.output_dir, pruning_method)
    write_pruning_summary(results, args.output_dir, pruning_method)

    print(f"[prune_mlp] saved pruning summary to {summary_path}")


def _load_weights(weights_path: Path) -> dict[str, np.ndarray]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    with np.load(weights_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _parse_prune_fractions(raw_fractions: str) -> list[float]:
    fractions = [float(x.strip()) for x in raw_fractions.split(",") if x.strip()]
    if not fractions:
        raise ValueError("At least one prune fraction must be provided.")
    invalid = [fraction for fraction in fractions if not 0.0 <= fraction <= 1.0]
    if invalid:
        raise ValueError(f"Prune fractions must be between 0.0 and 1.0. Invalid values: {invalid}")
    return fractions


def _evaluate_model(model, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(x)
    loss = model.loss.forward(logits, y)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))
    return float(loss), accuracy


def plot_pruning_tradeoffs(results: pd.DataFrame, output_dir: Path, pruning_method: str) -> None:
    method_label = pruning_method.capitalize()
    plt.figure(figsize=(8, 5))
    plt.plot(results["sparsity"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.plot(results["sparsity"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.title(f"{method_label} MLP pruning: parameter sparsity vs accuracy")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "pruning_sparsity_vs_accuracy.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["prune_fraction"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.plot(results["prune_fraction"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title(f"{method_label} MLP pruning: prune fraction vs accuracy")
    plt.xlabel("Prune fraction")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "pruning_fraction_vs_accuracy.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["sparsity"], results["test_loss"], marker="o", label="Test loss")
    plt.plot(results["sparsity"], results["val_loss"], marker="o", label="Validation loss")
    plt.title(f"{method_label} MLP pruning: parameter sparsity vs loss")
    plt.xlabel("Sparsity")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "pruning_sparsity_vs_loss.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    if "structured_sparsity" in results:
        plt.figure(figsize=(8, 5))
        plt.plot(results["structured_sparsity"], results["test_accuracy"], marker="o", label="Test accuracy")
        plt.plot(results["structured_sparsity"], results["val_accuracy"], marker="o", label="Validation accuracy")
        plt.title(f"{method_label} MLP pruning: hidden-neuron sparsity vs accuracy")
        plt.xlabel("Structured hidden-neuron sparsity")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "pruning_structured_sparsity_vs_accuracy.png", dpi=180)
        plt.close()


def write_pruning_summary(results: pd.DataFrame, output_dir: Path, pruning_method: str) -> None:
    method_label = "structured hidden-neuron pruning" if pruning_method == "structured" else "unstructured magnitude pruning"
    lines = [
        "# MLP Pruning Evaluation Summary",
        "",
        f"This report shows the effect of {method_label} on the manual MLP baseline.",
        "",
        "The magnitude-based path is kept in the code for optional comparison, but the WP4 result path is structured pruning.",
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

    lines.extend(
        [
            f"- Best validation accuracy: {best_val['val_accuracy']:.4f} at sparsity {best_val['sparsity']:.4f}",
            f"- Best test accuracy: {best_test['test_accuracy']:.4f} at sparsity {best_test['sparsity']:.4f}",
            "",
            "## Plots",
            "",
            "- `pruning_sparsity_vs_accuracy.png`",
            "- `pruning_fraction_vs_accuracy.png`",
            "- `pruning_sparsity_vs_loss.png`",
            "- `pruning_structured_sparsity_vs_accuracy.png`",
            "",
            "## Full results table",
            "",
            "| method | prune_fraction | parameter_sparsity | structured_sparsity | active_hidden_neurons | pruned_hidden_neurons | estimated_size_bytes | val_accuracy | test_accuracy | val_loss | test_loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in results.iterrows():
        lines.append(
            f"| {row['method']} | {row['prune_fraction']:.2f} | {row['sparsity']:.4f} | "
            f"{row['structured_sparsity']:.4f} | {int(row['active_hidden_neurons'])} | "
            f"{int(row['pruned_hidden_neurons'])} | {int(row['estimated_size_bytes'])} | "
            f"{row['val_accuracy']:.4f} | {row['test_accuracy']:.4f} | {row['val_loss']:.4f} | {row['test_loss']:.4f} |"
        )

    output_path = output_dir / "pruning_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
