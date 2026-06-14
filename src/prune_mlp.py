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
        description="Prune a trained manual MLP by magnitude and evaluate sparsity vs accuracy."
    )
    parser.add_argument("--weights", type=Path, default=Path("results/baseline_mlp/baseline_mlp_weights.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_mlp/pruning"))
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Use structured neuron pruning instead of unstructured weight pruning.",
    )
    parser.add_argument(
        "--prune-fractions",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,0.9",
        help="Comma-separated fractions of weights to prune (0.0-1.0).",
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

    prune_fractions = [float(x.strip()) for x in args.prune_fractions.split(",") if x.strip()]
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

        record = {
            "prune_fraction": prune_fraction,
            "sparsity": sparsity,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        records.append(record)

        prune_path = args.output_dir / f"pruned_{int(prune_fraction * 100):02d}.npz"
        np.savez(prune_path, **pruned_model.get_parameters())
        print(
            f"[prune_mlp] fraction={prune_fraction:.2f} sparsity={sparsity:.4f} "
            f"val_acc={val_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )

    results = pd.DataFrame(records)
    summary_path = args.output_dir / "pruning_metrics.csv"
    results.to_csv(summary_path, index=False)
    (args.output_dir / "pruning_metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    plot_pruning_tradeoffs(results, args.output_dir)
    write_pruning_summary(results, args.output_dir)

    print(f"[prune_mlp] saved pruning summary to {summary_path}")


def _load_weights(weights_path: Path) -> dict[str, np.ndarray]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    with np.load(weights_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _evaluate_model(model, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(x)
    loss = model.loss.forward(logits, y)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))
    return float(loss), accuracy


def plot_pruning_tradeoffs(results: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(results["sparsity"], results["val_accuracy"], marker="o", label="Validation accuracy")
    plt.plot(results["sparsity"], results["test_accuracy"], marker="o", label="Test accuracy")
    plt.title("Pruning sparsity vs accuracy")
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
    plt.title("Pruning fraction vs accuracy")
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
    plt.title("Pruning sparsity vs loss")
    plt.xlabel("Sparsity")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "pruning_sparsity_vs_loss.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()


def write_pruning_summary(results: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Pruning Evaluation Summary",
        "",
        "This report shows the effect of magnitude-based pruning on the manual MLP baseline.",
        "",
        "## Summary metrics",
        "",
        f"- Prune fractions: {', '.join(str(x) for x in results['prune_fraction'].tolist())}",
        f"- Sparsity values: {', '.join(f'{x:.4f}' for x in results['sparsity'].tolist())}",
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
            "",
            "## Full results table",
            "",
            "| prune_fraction | sparsity | val_accuracy | test_accuracy | val_loss | test_loss |",
            "|---|---|---|---|---|---|",
        ]
    )

    for _, row in results.iterrows():
        lines.append(
            f"| {row['prune_fraction']:.2f} | {row['sparsity']:.4f} | {row['val_accuracy']:.4f} | {row['test_accuracy']:.4f} | {row['val_loss']:.4f} | {row['test_loss']:.4f} |"
        )

    output_path = output_dir / "pruning_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
