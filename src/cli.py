from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.train_cnn import main as train_main
from src.train_mlp import main as train_mlp_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src", description="ECG model compression toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a baseline ECG classification model")
    train_parser.add_argument("--data-dir", type=str, default="data/processed")
    train_parser.add_argument("--output-dir", type=str, default="results/baseline_cnn")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--demo-data", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained Keras ECG model")
    evaluate_parser.add_argument("--model", type=str, default="results/baseline_cnn/tiny_ecg_cnn.keras")
    evaluate_parser.add_argument("--data-dir", type=str, default="data/processed")
    evaluate_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    evaluate_parser.add_argument("--demo-data", action="store_true")

    visualize_parser = subparsers.add_parser("visualize", help="Generate ECG dataset visualizations")
    visualize_parser.add_argument("--data-dir", type=str, default="data/processed")
    visualize_parser.add_argument("--output-dir", type=str, default="results/dataset_visualizations")
    visualize_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    visualize_parser.add_argument("--validation-fraction", type=float, default=0.15)
    visualize_parser.add_argument("--seed", type=int, default=42)
    visualize_parser.add_argument("--demo-data", action="store_true")
    visualize_parser.add_argument("--samples-per-class", type=int, default=3)

    train_mlp_parser = subparsers.add_parser("train-mlp", help="Train a baseline MLP ECG model")
    train_mlp_parser.add_argument("--data-dir", type=str, default="data/processed")
    train_mlp_parser.add_argument("--output-dir", type=str, default="results/baseline_mlp")
    train_mlp_parser.add_argument("--epochs", type=int, default=20)
    train_mlp_parser.add_argument("--batch-size", type=int, default=128)
    train_mlp_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_mlp_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_mlp_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    train_mlp_parser.add_argument("--seed", type=int, default=42)
    train_mlp_parser.add_argument("--hidden-units", type=int, default=64)
    train_mlp_parser.add_argument("--dense-layers", type=int, default=2)
    train_mlp_parser.add_argument("--quantize-aware", action="store_true", help="Enable quantization-aware training for the manual MLP.")
    train_mlp_parser.add_argument("--demo-data", action="store_true")

    prune_mlp_parser = subparsers.add_parser("prune-mlp", help="Prune a trained manual MLP and evaluate sparsity vs accuracy")
    prune_mlp_parser.add_argument("--weights", type=str, default="results/baseline_mlp/baseline_mlp_weights.npz")
    prune_mlp_parser.add_argument("--data-dir", type=str, default="data/processed")
    prune_mlp_parser.add_argument("--output-dir", type=str, default="results/baseline_mlp/pruning")
    prune_mlp_parser.add_argument("--demo-data", action="store_true")
    prune_mlp_parser.add_argument("--structured", action="store_true", help="Use structured neuron pruning instead of unstructured weight pruning.")
    prune_mlp_parser.add_argument("--prune-fractions", type=str, default="0.0,0.2,0.4,0.6,0.8,0.9")
    prune_mlp_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")

    prune_cnn_parser = subparsers.add_parser("prune-cnn", help="Prune the scratch CNN with structured compression and evaluate compression effects")
    prune_cnn_parser.add_argument("--weights", type=str, default="results/baseline_cnn/tiny_ecg_cnn_weights.npz")
    prune_cnn_parser.add_argument("--data-dir", type=str, default="data/processed")
    prune_cnn_parser.add_argument("--output-dir", type=str, default="results/baseline_cnn/pruning")
    prune_cnn_parser.add_argument("--demo-data", action="store_true")
    prune_cnn_parser.add_argument("--batch-size", type=int, default=128)
    prune_cnn_parser.add_argument("--magnitude", action="store_true", help="Use unstructured magnitude pruning instead of the default structured CNN pruning.")
    prune_cnn_parser.add_argument("--prune-fractions", type=str, default="0.0,0.2,0.4,0.6,0.8,0.9")
    prune_cnn_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")

    quantize_mlp_parser = subparsers.add_parser("quantize-mlp", help="Quantize the manual MLP to 8-bit fixed-point and compare to the original model")
    quantize_mlp_parser.add_argument("--weights", type=str, default="results/baseline_mlp/baseline_mlp_weights.npz")
    quantize_mlp_parser.add_argument("--data-dir", type=str, default="data/processed")
    quantize_mlp_parser.add_argument("--output-dir", type=str, default="results/baseline_mlp/quantization")
    quantize_mlp_parser.add_argument("--demo-data", action="store_true")
    quantize_mlp_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    quantize_mlp_parser.add_argument("--calibration-samples", type=int, default=200)

    quantize_parser = subparsers.add_parser("quantize", help="Convert a trained model to int8 TensorFlow Lite")
    quantize_parser.add_argument("--model", type=str, default="results/baseline_cnn/tiny_ecg_cnn.keras")
    quantize_parser.add_argument("--data-dir", type=str, default="data/processed")
    quantize_parser.add_argument("--output", type=str, default="results/quantized/tiny_ecg_cnn_int8.tflite")
    quantize_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    quantize_parser.add_argument("--representative-samples", type=int, default=200)
    quantize_parser.add_argument("--demo-data", action="store_true")

    summarize_parser = subparsers.add_parser("summarize", help="Generate evaluation plots and a markdown summary")
    summarize_parser.add_argument("--run-dir", type=str, default="results/baseline_cnn")
    summarize_parser.add_argument(
        "--quantized-dir",
        type=str,
        default=None,
        help="Optional directory containing int8_metrics.json for baseline vs quantized comparison.",
    )

    return parser.parse_args()


def _build_forwarded_argv(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [sys.argv[0]]
    for key, value in vars(args).items():
        if key == "command" or value is False or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        forwarded.append(flag)
        if not isinstance(value, bool):
            forwarded.append(str(value))
    return forwarded


def main() -> None:
    args = parse_args()
    sys.argv = _build_forwarded_argv(args)

    if args.command == "train":
        train_main()
    elif args.command == "evaluate":
        from src.evaluate_model import main as evaluate_main

        evaluate_main()
    elif args.command == "visualize":
        from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
        from src.data.visualization import visualize_dataset

        dataset = (
            make_demo_dataset(
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            if args.demo_data
            else load_mitbih_csv(
                data_dir=Path(args.data_dir),
                validation_fraction=args.validation_fraction,
                normalize=args.normalize,
                seed=args.seed,
            )
        )
        visualize_dataset(
            dataset,
            Path(args.output_dir),
            samples_per_class=args.samples_per_class,
        )
    elif args.command == "train-mlp":
        train_mlp_main()
    elif args.command == "prune-mlp":
        from src.prune_mlp import main as prune_mlp_main

        prune_mlp_main()
    elif args.command == "prune-cnn":
        from src.prune_cnn import main as prune_cnn_main

        prune_cnn_main()
    elif args.command == "quantize-mlp":
        from src.quantize_mlp import main as quantize_mlp_main

        quantize_mlp_main()
    elif args.command == "quantize":
        from src.compression.quantize_tflite import main as quantize_main

        quantize_main()
    elif args.command == "summarize":
        from src.evaluation.summarize_baseline import main as summarize_main

        summarize_main()
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
