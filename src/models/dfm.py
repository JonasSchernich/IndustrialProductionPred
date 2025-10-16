
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import Ridge

class ForecastModel:
    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp.copy()
        alpha = self.hp.get("alpha", 0.0)
        # Ridge mit kleinem alpha ≈ OLS; robuster bei p>>n
        self.model = Ridge(alpha=alpha, fit_intercept=True, random_state=self.hp.get("seed", 123))

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        self.model.fit(X_tr, y_tr)

    def predict_one(self, x_eval: np.ndarray) -> float:
        yhat = self.model.predict(x_eval)
        return float(yhat[0])

    def get_name(self) -> str:
        return "dfm"
