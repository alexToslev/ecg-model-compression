from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Keras ECG model.")
    parser.add_argument("--model", type=Path, default=Path("results/baseline_cnn/baseline_ecg_cnn.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--demo-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = make_demo_dataset() if args.demo_data else load_mitbih_csv(args.data_dir, normalize=args.normalize)
    model = tf.keras.models.load_model(args.model)
    loss, accuracy = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    predictions = model.predict(dataset.x_test, verbose=0).argmax(axis=1)

    print(json.dumps({"loss": float(loss), "accuracy": float(accuracy)}, indent=2))
    print(classification_report(dataset.y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
