# src/models/Baselines.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


class ForecastModel:
    """
    Implementiert Baseline-Modelle (RW, Mean), die die X-Features ignorieren.
    - .fit(X, y, sample_weight=None)
    - .predict_one(x_row)
    - HPs:
        - "model_type": "RW" oder "Mean"
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
        Passt das Modell an. X wird ignoriert.
        y ist y_tr aus der Pipeline (d.h. y.shift(-1).iloc[taus_model]).
        """
        if self.model_type == "Mean":
            # Berechnet den Expanding Grand Mean
            # y ist hier der Vektor der Zielvariablen im Trainingsfenster
            if sample_weight is not None:
                self._mean_pred = np.average(y, weights=sample_weight)
            else:
                self._mean_pred = np.mean(y)

        elif self.model_type == "RW":
            # RW (Prognose 0) erfordert kein Training
            pass

        else:
            raise ValueError(f"Unbekannter model_type in Baselines: {self.model_type}")

        return self

    def predict(self, X):
        """Erstellt Prognosen für einen Batch."""
        if self.model_type == "Mean":
            return np.full(len(X), self._mean_pred)
        elif self.model_type == "RW":
            return np.zeros(len(X))

    def predict_one(self, x_row):
        """Erstellt eine einzelne Prognose."""
        if self.model_type == "Mean":
            return self._mean_pred
        elif self.model_type == "RW":
            return 0.0