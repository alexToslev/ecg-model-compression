from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="ECG CNN improvement toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the improved class-weighted ECG CNN",
    )
    train_parser.add_argument("--data-dir", type=str, default="data/processed")
    train_parser.add_argument("--output-dir", type=str, default="results/improved_cnn_scratch")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    train_parser.add_argument("--dropout-rate", type=float, default=0.2)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--demo-data", action="store_true")
    train_parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    train_parser.add_argument("--class-weight-cap", type=float, default=10.0)
    train_parser.add_argument("--sampling-strategy", choices=["shuffle", "weighted", "balanced"], default="shuffle")
    train_parser.add_argument("--augment-rare-classes", dest="augment_rare_classes", action="store_true", default=True)
    train_parser.add_argument("--no-augment-rare-classes", dest="augment_rare_classes", action="store_false")
    train_parser.add_argument("--rare-target-count", type=int, default=2000)
    train_parser.add_argument("--rare-threshold", type=float, default=0.2)
    train_parser.add_argument("--max-train-samples", type=int, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained Keras ECG model")
    evaluate_parser.add_argument("--model", type=str, default="results/improved_cnn_scratch/tiny_ecg_cnn.keras")
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

    quantize_parser = subparsers.add_parser("quantize", help="Convert the improved CNN to int8 TensorFlow Lite")
    quantize_parser.add_argument("--model", type=str, default="results/improved_cnn_scratch/tiny_ecg_cnn.keras")
    quantize_parser.add_argument("--data-dir", type=str, default="data/processed")
    quantize_parser.add_argument("--output", type=str, default="results/improved_cnn_scratch/tiny_ecg_cnn_int8.tflite")
    quantize_parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    quantize_parser.add_argument("--representative-samples", type=int, default=200)
    quantize_parser.add_argument("--demo-data", action="store_true")

    tinyml_parser = subparsers.add_parser(
        "export-tinyml",
        help="Create an ESP32/TinyML export package and deployment simulation report",
    )
    tinyml_parser.add_argument("--model", type=str, default="results/improved_cnn_scratch/tiny_ecg_cnn_int8.tflite")
    tinyml_parser.add_argument("--metrics", type=str, default="results/improved_cnn_scratch/int8_metrics.json")
    tinyml_parser.add_argument("--output-dir", type=str, default="results/esp32")
    tinyml_parser.add_argument("--model-name", type=str, default="improved_ecg_cnn_int8")
    tinyml_parser.add_argument("--arena-bytes", type=int, default=None)
    tinyml_parser.add_argument("--flash-budget-bytes", type=int, default=4 * 1024 * 1024)
    tinyml_parser.add_argument("--sram-budget-bytes", type=int, default=320 * 1024)

    samples_parser = subparsers.add_parser(
        "export-ecg-samples",
        help="Export real MIT-BIH test beats for ESP32 int8 inference",
    )
    samples_parser.add_argument("--csv", type=str, default="data/processed/mitbih_test.csv")
    samples_parser.add_argument("--output-dir", type=str, default="results/esp32_final_cap_4")
    samples_parser.add_argument("--report", type=str, default="results/esp32_final_cap_4/esp32_deployment_report.json")
    samples_parser.add_argument("--model", type=str, default="results/final_candidate_cap_4/tiny_ecg_cnn_int8.tflite")
    samples_parser.add_argument("--samples-per-class", type=int, default=1)
    samples_parser.add_argument("--seed", type=int, default=42)
    samples_parser.add_argument("--prefer-correct", action="store_true")
    samples_parser.add_argument("--input-scale", type=float, default=None)
    samples_parser.add_argument("--input-zero-point", type=int, default=None)
    samples_parser.add_argument("--model-header", type=str, default="model.h")
    samples_parser.add_argument("--sample-header", type=str, default="ecg_samples.h")
    samples_parser.add_argument("--model-array-name", type=str, default="g_model")

    summarize_parser = subparsers.add_parser("summarize", help="Generate evaluation plots and a markdown summary")
    summarize_parser.add_argument("--run-dir", type=str, default="results/improved_cnn_scratch")
    summarize_parser.add_argument("--quantized-dir", type=str, default=None)

    final_compare_parser = subparsers.add_parser(
        "final-compare",
        help="Create final cap-4 quantized plots and baseline-vs-INT8 comparison plots",
    )
    final_compare_parser.add_argument("--baseline-dir", type=str, default="results/class_weight_sweep_correct_data/cap_4")
    final_compare_parser.add_argument("--int8-dir", type=str, default="results/final_candidate_cap_4")
    final_compare_parser.add_argument("--output-dir", type=str, default="results/final_cap4_comparison")
    final_compare_parser.add_argument("--esp32-latency-ms", type=float, default=52.0)
    final_compare_parser.add_argument("--tensor-arena-kib", type=float, default=80.0)

    return parser.parse_args()


def forwarded_argv(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [sys.argv[0]]
    for key, value in vars(args).items():
        if key == "command" or value is None:
            continue
        if key == "augment_rare_classes" and value is False:
            forwarded.append("--no-augment-rare-classes")
            continue
        if value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        forwarded.append(flag)
        if not isinstance(value, bool):
            forwarded.append(str(value))
    return forwarded


def main() -> None:
    args = parse_args()
    sys.argv = forwarded_argv(args)

    if args.command == "train":
        from src.train_cnn import main as train_main

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
        visualize_dataset(dataset, Path(args.output_dir), samples_per_class=args.samples_per_class)
    elif args.command == "quantize":
        from src.compression.quantize_tflite import main as quantize_main

        quantize_main()
    elif args.command == "export-tinyml":
        from src.export_tinyml import main as export_tinyml_main

        export_tinyml_main()
    elif args.command == "export-ecg-samples":
        from src.export_ecg_samples import main as export_ecg_samples_main

        export_ecg_samples_main()
    elif args.command == "summarize":
        from src.evaluation.summarize_baseline import main as summarize_main

        summarize_main()
    elif args.command == "final-compare":
        from src.evaluation.final_quantized_comparison import main as final_compare_main

        final_compare_main()


if __name__ == "__main__":
    main()
