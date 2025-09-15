"""Feature engineering module for product weight and volume prediction."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def safe_parse(x) -> dict | None:
    """Parse JSON string if value is not NA."""
    if pd.notna(x):
        return json.loads(x)
    return None


def get_weight_or_vol(values: dict, search_key: str) -> str | None:
    """Search for weight or volume label in list of dictionaries."""
    try:
        return next(d["value"] for d in values if d.get("name") == search_key)
    except (StopIteration, TypeError):
        return None


def to_kg(value: float, unit: str) -> float | None:
    """Convert weight values to kilograms."""
    if pd.isna(value) or pd.isna(unit):
        return None

    unit = unit.lower()
    if unit == "kg":
        return value
    elif unit == "lb":
        return value * 0.4536
    elif unit == "oz":
        return value * 0.0283
    return None


def extract_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract volume information from product specifications."""
    col_name = "specifications_parsed"
    search_key = "Assembled Product Dimensions (L x W x H)"
    col_l, col_w, col_h, col_units = "size_l", "size_w", "size_h", "size_units"

    vols = df.copy()
    vols = vols.set_index(keys=["sku"])

    # Extract volume info
    vols_sr = vols[col_name].apply(
        lambda x: get_weight_or_vol(x, search_key=search_key)
    )

    # Parse dimensions
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


def extract_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Extract weight information from product specifications."""
    col_name = "specifications_parsed"
    search_key = "Assembled Product Weight"
    col_weight = "weight"
    col_weight_unit = "weight_unit"

    weights = df.copy()
    weights = weights.set_index(keys=["sku"])

    # Extract weight info
    weights_sr = weights[col_name].apply(
        lambda x: get_weight_or_vol(x, search_key=search_key)
    )

    # Parse weights
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


def calculate_mean_metrics(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Calculate mean weights and volumes by category."""
    mean_volumes = (
        df.groupby(["root_category_name"])
        .agg(
            {
                "size_l": "mean",
                "size_w": "mean",
                "size_h": "mean",
            }
        )
        .dropna()
        .round(1)
        .to_dict(orient="index")
    )

    mean_weights = (
        df.groupby(["root_category_name"])
        .agg({"weight_kg": "mean"})
        .dropna()
        .round(2)
        .to_dict(orient="index")
    )

    return mean_weights, mean_volumes


def fill_missing_values(
    df: pd.DataFrame, mean_weights: Dict, mean_volumes: Dict
) -> pd.DataFrame:
    """Fill missing values using category means."""

    def fill_weight(row):
        cat = row["root_category_name"]
        mean = mean_weights.get(cat, {}).get("weight_kg")
        return mean if mean is not None else None

    def fill_volume(row):
        cat = row["root_category_name"]
        mean = mean_volumes.get(cat, {})
        if mean:
            return pd.Series(
                {
                    "size_l_mean": mean.get("size_l"),
                    "size_w_mean": mean.get("size_w"),
                    "size_h_mean": mean.get("size_h"),
                }
            )
        return pd.Series(
            [None, None, None], index=["size_l_mean", "size_w_mean", "size_h_mean"]
        )

    df["weight_kg_mean"] = df.apply(fill_weight, axis=1)
    df[["size_l_mean", "size_w_mean", "size_h_mean"]] = df.apply(fill_volume, axis=1)

    # Fix specific weight issues (800 mg case)
    df["weight_kg"] = np.where(df.eval("weight_kg==0"), 0.8, df["weight_kg"])

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature processing pipeline."""
    # Parse JSON columns
    df["specifications_parsed"] = df["specifications"].apply(safe_parse)
    df["other_attributes_parsed"] = df["other_attributes"].apply(safe_parse)

    # Extract volumes and merge
    volumes_df = extract_volumes(df)
    df = df.merge(volumes_df, how="left", on="sku")

    # Extract weights and merge
    weights_df = extract_weights(df)
    df = df.merge(weights_df, how="left", on="sku")

    # Convert weights to kg
    df["weight_kg"] = df.apply(
        lambda row: to_kg(row["weight"], row["weight_unit"]), axis=1
    )

    # Calculate means and fill missing values
    mean_weights, mean_volumes = calculate_mean_metrics(df)
    df = fill_missing_values(df, mean_weights, mean_volumes)

    return df
