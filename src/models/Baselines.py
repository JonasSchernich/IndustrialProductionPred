# src/models/Baselines.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


class ForecastModel:
    """
    Baseline models (RW, Mean) that ignore X features.
    - .fit(X, y, sample_weight=None)
    - .predict_one(x_row)
    - Hyperparameters:
        - "model_type": "RW" or "Mean"
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self.model_type = str(self.params.get("model_type", "RW"))
        self._mean_pred = 0.0
        self._backend_name = f"baseline_{self.model_type}"

    def get_name(self) -> str:
        return self._backend_name

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model. X is ignored.
        y is the training target vector from the pipeline (e.g., y.shift(-1).iloc[taus_model]).
        """
        if self.model_type == "Mean":
            # Compute (weighted) mean over the training window
            if sample_weight is not None:
                self._mean_pred = np.average(y, weights=sample_weight)
            else:
                self._mean_pred = np.mean(y)

        elif self.model_type == "RW":
            # RW baseline (predict 0) requires no training
            pass

        else:
            raise ValueError(f"Unknown model_type in Baselines: {self.model_type}")

        return self

    def predict(self, X):
        """Create predictions for a batch."""
        if self.model_type == "Mean":
            return np.full(len(X), self._mean_pred)
        elif self.model_type == "RW":
            return np.zeros(len(X))

    def predict_one(self, x_row):
        """Create a single prediction."""
        if self.model_type == "Mean":
            return self._mean_pred
        elif self.model_type == "RW":
            return 0.0
