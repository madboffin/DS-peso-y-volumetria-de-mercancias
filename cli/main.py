import argparse
import sys
from pathlib import Path
from typing import Optional


def setup_cli_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Product Weight & Volume Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="{train,predict}"
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


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print("No command provided.")
        return 1

    if args.command == "predict":
        print(args)
        return 0


if __name__ == "__main__":
    print("This script is intended to be run as a module.")
    sys.exit(main())
