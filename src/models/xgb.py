
from __future__ import annotations
from typing import Dict, Any
import numpy as np

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingRegressor

class ForecastModel:
    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp.copy()
        self.model = None

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        if _HAS_XGB:
            params = dict(
                objective="reg:squarederror",
                learning_rate=self.hp.get("learning_rate", 0.05),
                max_depth=self.hp.get("max_depth", 3),
                subsample=self.hp.get("subsample", 0.8),
                colsample_bytree=self.hp.get("colsample_bytree", 0.8),
                reg_lambda=self.hp.get("reg_lambda", 0.0),
                reg_alpha=self.hp.get("reg_alpha", 0.0),
                n_estimators=self.hp.get("n_estimators", 200),
                random_state=self.hp.get("seed", 123),
            )
            self.model = XGBRegressor(**params)
        else:
            params = dict(
                learning_rate=self.hp.get("learning_rate", 0.05),
                max_depth=self.hp.get("max_depth", None),
                max_iter=self.hp.get("n_estimators", 200),
                l2_regularization=self.hp.get("reg_lambda", 0.0)
            )
            self.model = HistGradientBoostingRegressor(**params)
        self.model.fit(X_tr, y_tr)

    def predict_one(self, x_eval: np.ndarray) -> float:
        yhat = self.model.predict(x_eval)
        return float(yhat[0])

    def get_name(self) -> str:
        return "xgb" if _HAS_XGB else "hgb_fallback"
