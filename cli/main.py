from pathlib import Path
from typing import Optional
import argparse
import sys

import pandas as pd

from src.data import feature_engineering as fe


def preprocess_data(input_path: Path, output_dir: Path) -> int:
    """Preprocess raw data and save results."""
    try:
        # Read input data
        dtypes = {"sku": str, "gtin": str}
        df = pd.read_csv(input_path, dtype=dtypes)

        # Process features
        df_processed = fe.process_features(df)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed dataframe
        output_file = output_dir / "processed_data.csv"
        df_processed.to_csv(output_file, index=False, sep="|")

        return 0

    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        return 1


def setup_cli_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Product Weight & Volume Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="{predict}"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make predictions on new data",
        description="Predict weight and dimensions for new products",
    )
    predict_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input CSV file with products to predict",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output CSV file for predictions",
    )

    return parser


def predict(input_path: Path, output_path: Path) -> int:
    """Make predictions on new data."""
    try:
        # TODO: Implement prediction logic
        print("Prediction functionality not yet implemented")
        return 0

    except Exception as e:
        print(f"Error making predictions: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print("No command provided.")
        return 1

    if args.command == "predict":
        try:
            output_dir = args.output.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            processed_data_dir = "../data/processed"
            preprocess_data(args.input, processed_data_dir)
            predict(args.input, args.output)
            return 0
        except Exception as e:
            print(f"Error processing data: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
