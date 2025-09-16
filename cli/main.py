from pathlib import Path
from typing import Optional
import argparse
import sys

import pandas as pd

from src.preprocessing import feature_engineering as fe
from src.models.predictor import ProductPredictor


def preprocess_data(input_path: Path, output_dir: Path):
    """Preprocess raw data and save results."""
    try:
        dtypes = {"sku": str, "gtin": str}
        df = pd.read_csv(input_path, dtype=dtypes)
        return fe.process_features(df)

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


def predict(df):
    """Make predictions on new data."""
    try:
        # Load model and make predictions
        model_dir = Path().cwd() / "models" / "trained_models"
        predictor = ProductPredictor(model_dir)
        return predictor.predict(df)

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
            processed_data_dir = Path("../data/processed")
            df = preprocess_data(args.input, processed_data_dir)
            predictions = predict(df)
            return 0
        except Exception as e:
            print(f"Error processing data: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
