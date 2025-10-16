# src/lgbm.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception as e:
    raise RuntimeError("LightGBM is required but not installed. Please install `lightgbm`.") from e


class ForecastModel:
    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp.copy()
        self.model = None
        self.backend = "lightgbm"

    def _split_train_dev(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Zeitbewusst: Dev-Tail ist das jüngste Ende des Trainingsfensters.
        """
        n = X.shape[0]
        dev = max(8, min(24, max(1, int(0.1 * n))))
        if n <= dev + 10:
            return X, y, None, None
        return X[:-dev], y[:-dev], X[-dev:], y[-dev:]

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        Xb_tr, yb_tr, Xb_dev, yb_dev = self._split_train_dev(X_tr, y_tr)
        params: Dict[str, Any] = dict(
            learning_rate=self.hp.get("learning_rate", 0.05),
            n_estimators=self.hp.get("n_estimators", 300),
            subsample=self.hp.get("subsample", 0.8),
            colsample_bytree=self.hp.get("colsample_bytree", 0.8),
            reg_lambda=self.hp.get("reg_lambda", 5.0),
            reg_alpha=self.hp.get("reg_alpha", 0.0),
            max_depth=self.hp.get("max_depth", -1),
            random_state=self.hp.get("seed", 123),
            objective="regression",
            n_jobs=self.hp.get("n_jobs", 0),
            max_bin=self.hp.get("max_bin", 63),
            min_data_in_bin=self.hp.get("min_data_in_bin", 1),
            min_gain_to_split=self.hp.get("min_gain_to_split", 0.0),
        )
        # optional & versionssicher setzen
        ffbn = self.hp.get("feature_fraction_bynode", None)
        if ffbn is not None:
            params["feature_fraction_bynode"] = ffbn

        model = LGBMRegressor(**params)
        if Xb_dev is not None:
            model.fit(
                Xb_tr, yb_tr,
                eval_set=[(Xb_dev, yb_dev)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model.fit(Xb_tr, yb_tr)
        self.model = model

    def predict_one(self, x_eval: np.ndarray) -> float:
        yhat = self.model.predict(x_eval)
        return float(yhat[0])

    def get_name(self) -> str:
        return f"lgbm[{self.backend}]"
