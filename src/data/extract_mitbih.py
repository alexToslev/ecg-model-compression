from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MIT-BIH CSV dataset files from a zip archive.")
    parser.add_argument(
        "--zip-file",
        type=Path,
        default=None,
        help="Path to the zip archive containing mitbih_train.csv and mitbih_test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where extracted CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.zip_file or _find_zip_archive()
    if zip_path is None:
        raise FileNotFoundError(
            "No zip archive was provided and none was found in the repository root. "
            "Provide --zip-file path/to/archive.zip."
        )

    with zipfile.ZipFile(zip_path, "r") as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV files found inside {zip_path}")

        extracted = []
        for member in csv_files:
            destination = args.output_dir / Path(member).name
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            extracted.append(destination.name)

    print(f"Extracted {len(extracted)} CSV file(s) to {args.output_dir}")
    for name in extracted:
        print(f" - {name}")


def _find_zip_archive() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    zip_files = sorted(root.glob("*.zip"))
    return zip_files[0] if zip_files else None


if __name__ == "__main__":
    main()
