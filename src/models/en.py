
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import ElasticNet

class ForecastModel:
    """
    Unified Elastic Net model.
    Any dimensionality reduction (PCA or PLS) is controlled outside via cfg.dr_method
    and applied in features.fit_dr / features.transform_dr.
    """
    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp.copy()
        alpha = float(self.hp.get("alpha", 0.001))
        l1_ratio = float(self.hp.get("l1_ratio", 0.2))
        max_iter = int(self.hp.get("max_iter", 5000))
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            fit_intercept=True,
            random_state=self.hp.get("seed", 123)
        )

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        self.model.fit(X_tr, y_tr)

    def predict_one(self, x_eval: np.ndarray) -> float:
        yhat = self.model.predict(x_eval)
        return float(yhat[0])

    def get_name(self) -> str:
        return "elastic_net"
