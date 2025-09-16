"""Model prediction module."""

from pathlib import Path
import joblib

import pandas as pd


class ProductPredictor:
    def __init__(self, model_dir: Path):
        """Initialize predictor with model directory.

        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
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

        # TODO implement
        return None
