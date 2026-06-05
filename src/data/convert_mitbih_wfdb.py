from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import wfdb

LABEL_MAP = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "V": 2,
    "E": 2,
    "F": 3,
    "Q": 4,
    "P": 4,
    "/": 4,
    "f": 4,
    "x": 4,
}

WINDOW_SIZE = 187
PRE_SAMPLES = 93


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw MIT-BIH WFDB records into preprocessed heartbeat CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("mit-bih-arrhythmia-database-1.0.0"),
        help="Root directory containing MIT-BIH .hea/.dat/.atr files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write mitbih_train.csv and mitbih_test.csv.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of records used for training; the remainder are used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling record order.",
    )
    parser.add_argument(
        "--min-beats",
        type=int,
        default=0,
        help="Minimum number of valid beats required for a record to be included.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_names = _find_record_names(args.input_dir)
    if not record_names:
        raise FileNotFoundError(
            f"No WFDB .hea files found under {args.input_dir}."
        )

    np.random.seed(args.seed)
    np.random.shuffle(record_names)

    split = int(len(record_names) * args.train_fraction)
    train_records = sorted(record_names[:split])
    test_records = sorted(record_names[split:])

    train_rows, train_counts = _process_records(args.input_dir, train_records, args.min_beats)
    test_rows, test_counts = _process_records(args.input_dir, test_records, args.min_beats)

    _write_csv(args.output_dir / "mitbih_train.csv", train_rows)
    _write_csv(args.output_dir / "mitbih_test.csv", test_rows)

    print("Wrote dataset files to:")
    print(f"  {args.output_dir / 'mitbih_train.csv'} ({len(train_rows)} rows)")
    print(f"  {args.output_dir / 'mitbih_test.csv'} ({len(test_rows)} rows)")
    print("Label distribution:")
    print(f"  train: {train_counts}")
    print(f"  test:  {test_counts}")


def _find_record_names(input_dir: Path) -> list[str]:
    heas = sorted(input_dir.glob("*.hea"))
    return [hea.stem for hea in heas]


def _process_records(input_dir: Path, record_names: list[str], min_beats: int) -> tuple[list[list[float]], dict[int, int]]:
    rows: list[list[float]] = []
    counts: dict[int, int] = {}
    for record_name in record_names:
        record_path = input_dir / record_name
        try:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), "atr")
        except Exception as exc:
            print(f"Skipping record {record_name}: failed to read ({exc})")
            continue

        signal = record.p_signal[:, 0].astype(np.float32)
        beats = _extract_labelled_beats(signal, annotation)
        if len(beats) < min_beats:
            print(f"Skipping record {record_name}: only {len(beats)} valid beats")
            continue

        for segment, label in beats:
            rows.append([*segment.tolist(), label])
            counts[label] = counts.get(label, 0) + 1

    return rows, counts


def _extract_labelled_beats(signal: np.ndarray, annotation) -> list[tuple[np.ndarray, int]]:
    beats: list[tuple[np.ndarray, int]] = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if isinstance(symbol, bytes):
            symbol = symbol.decode("utf-8", errors="ignore")
        if symbol not in LABEL_MAP:
            continue
        label = LABEL_MAP[symbol]
        segment = _extract_segment(signal, sample)
        if segment is None:
            continue
        beats.append((segment, label))
    return beats


def _extract_segment(signal: np.ndarray, center_index: int) -> np.ndarray | None:
    start = center_index - PRE_SAMPLES
    end = start + WINDOW_SIZE
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def _write_csv(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
