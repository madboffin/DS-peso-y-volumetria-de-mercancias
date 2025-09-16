"""Model prediction module."""

from pathlib import Path
import joblib
import numpy as np
import spacy

import pandas as pd


class ProductPredictor:
    def __init__(self, model_dir: Path):
        """Initialize predictor with model directory.

        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.nlp = spacy.load("en_core_web_md")
        self.nlp.disable_pipes("parser", "ner")
        self._load_models()

    def _load_models(self):
        """Load all saved models from model directory."""
        model_files = {
            "weight": "weight_model_svr.pkl",
            "size_l": "size_l_svr.pkl",
            "size_w": "size_w_svr.pkl",
            "size_h": "size_h_svr.pkl",
        }

        for target, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                self.models[target] = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for weight and volume.

        Args:
            df: Input dataframe with product information

        Returns:
            DataFrame with original data and predictions added
        """
        if not self.models:
            raise ValueError("No models loaded. Check model directory.")

        # Create a copy to avoid modifying input
        df_out = df.copy()

        # Generate embeddings for product names
        df_out["product_name_vector"] = df_out["product_name"].apply(
            lambda x: self.nlp(x).vector
        )
        vectors = np.stack(df_out["product_name_vector"].values)

        # Predict weight if missing
        weight_mask = df_out["weight_kg"].isna()
        if weight_mask.any() and "weight" in self.models:
            weight_pred = self.models["weight"].predict(vectors[weight_mask])
            df_out.loc[weight_mask, "weight_kg_predicted"] = weight_pred

        # Predict dimensions if missing
        size_mask = df_out[["size_l", "size_w", "size_h"]].isna().all(axis=1)
        if size_mask.any():
            for dim in ["size_l", "size_w", "size_h"]:
                if dim in self.models:
                    dim_pred = self.models[dim].predict(vectors[size_mask])
                    df_out.loc[size_mask, f"{dim}_predicted"] = dim_pred

        # Drop the vector column as it's no longer needed
        df_out.drop(columns=["product_name_vector"], inplace=True)

        return df_out
