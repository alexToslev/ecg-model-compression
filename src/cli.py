from __future__ import annotations

import argparse
import sys

from src.compression.quantize_tflite import main as quantize_main
from src.evaluate_model import main as evaluate_main
from src.train_cnn import main as train_main
from src.evaluation.summarize_baseline import main as summarize_main


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

    quantize_parser = subparsers.add_parser("quantize", help="Convert a trained model to int8 TensorFlow Lite")
    quantize_parser.add_argument("--model", type=str, default="results/baseline_cnn/tiny_ecg_cnn.keras")
    quantize_parser.add_argument("--data-dir", type=str, default="data/processed")
    quantize_parser.add_argument("--output", type=str, default="results/quantized/tiny_ecg_cnn_int8.tflite")
    quantize_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    quantize_parser.add_argument("--representative-samples", type=int, default=200)
    quantize_parser.add_argument("--demo-data", action="store_true")

    summarize_parser = subparsers.add_parser("summarize", help="Generate evaluation plots and a markdown summary")
    summarize_parser.add_argument("--run-dir", type=str, default="results/baseline_cnn")

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
    if args.command == "train":
        sys.argv = _build_forwarded_argv(args)
        train_main()
    elif args.command == "evaluate":
        sys.argv = _build_forwarded_argv(args)
        evaluate_main()
    elif args.command == "quantize":
        sys.argv = _build_forwarded_argv(args)
        quantize_main()
    elif args.command == "summarize":
        sys.argv = _build_forwarded_argv(args)
        summarize_main()
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
