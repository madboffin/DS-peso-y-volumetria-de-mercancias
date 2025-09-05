import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def safe_parse(x):
    """Parse JSON string if value is not NA."""
    if pd.notna(x):
        return json.loads(x)
    return None


def get_weight_or_vol(values: dict, search_key) -> str | None:
    """Search for weight or volume label in list of dictionaries."""
    try:
        return next(d["value"] for d in values if d.get("name") == search_key)
    except (StopIteration, TypeError):
        return None


def process_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process volume information."""
    col_name = "specifications_parsed"
    search_key = "Assembled Product Dimensions (L x W x H)"

    # Create copy and set index
    vols = df.copy()
    vols = vols.set_index(keys=["sku"])

    # Extract volume info
    vols_sr = vols[col_name].apply(
        lambda x: get_weight_or_vol(x, search_key=search_key)
    )

    # Parse dimensions
    col_l, col_w, col_h, col_units = "size_l", "size_w", "size_h", "size_units"
    vols = vols_sr.str.extractall(
        rf"(?P<{col_l}>[\d\.]+) x (?P<{col_w}>[\d\.]+) x (?P<{col_h}>[\d\.]+) (?P<{col_units}>\w+)"
    )
    vols = vols.droplevel(level=1)  # Remove match index

    # Convert to numeric
    for col in [col_h, col_w, col_l]:
        vols[col] = pd.to_numeric(vols[col])

    # Convert feet to inches
    vols.loc[vols[col_units] == "Feet", [col_h, col_w, col_l]] *= 12
    vols[col_units] = "in"

    return vols.reset_index()


def process_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process weight information."""
    col_name = "specifications_parsed"
    search_key = "Assembled Product Weight"
    col_weight = "weight_value"
    col_weight_unit = "weight_unit"

    # Create copy and set index
    weights = df.copy()
    weights = weights.set_index(keys=["sku"])

    # Extract weight info
    weights_sr = weights[col_name].apply(
        lambda x: get_weight_or_vol(x, search_key=search_key)
    )

    # Parse weights and units
    weights = weights_sr.str.extractall(
        rf"(?P<{col_weight}>[\d\.]+) ?(?P<{col_weight_unit}>\w+)"
    )
    weights[col_weight] = pd.to_numeric(weights[col_weight])
    weights[col_weight_unit] = weights[col_weight_unit].str.lower()

    # Normalize units
    normalize_units = {
        "lbs": "lb",
        "pounds": "lb",
        "ounces": "oz",
    }
    weights[col_weight_unit] = weights[col_weight_unit].replace(normalize_units)
    weights = weights.droplevel(level=1)

    return weights.reset_index()


def process_data(input_path: Path) -> pd.DataFrame:
    """Main data processing pipeline."""
    # Read input data
    dtypes = {"sku": str, "gtin": str}
    df = pd.read_csv(input_path, dtype=dtypes)

    # Parse JSON columns
    df["specifications_parsed"] = df["specifications"].apply(safe_parse)
    df["other_attributes_parsed"] = df["other_attributes"].apply(safe_parse)

    # Process volumes and weights
    df = df.merge(process_volumes(df), how="left", on="sku")
    df = df.merge(process_weights(df), how="left", on="sku")

    return df


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
        try:
            output_dir = args.output.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            df = process_data(args.input)
            df.to_csv(args.output, index=False)
            return 0
        except Exception as e:
            print(f"Error processing data: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
