
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

def _make_member(kind: str, hp: Dict[str, Any]):
    if kind == "lgbm":
        from .lgbm import ForecastModel as M
        return M(hp)
    elif kind == "xgb":
        from .xgb import ForecastModel as M
        return M(hp)
    elif kind == "en_pca":
        from .en_pca import ForecastModel as M
        return M(hp)
    elif kind == "pls_en":
        from .pls_en import ForecastModel as M
        return M(hp)
    elif kind == "dfm":
        from .dfm import ForecastModel as M
        return M(hp)
    else:
        raise ValueError(f"Unknown member type: {kind}")

class ForecastModel:
    def __init__(self, hp: Dict[str, Any]):
        """
        hp example:
        {
          "members": [
             {"type":"lgbm","hp":{...}},
             {"type":"xgb","hp":{...}}
          ],
          "weights": None  # or list of floats
        }
        """
        self.hp = hp.copy()
        members_spec = self.hp.get("members", [])
        if not members_spec:
            raise ValueError("Ensemble requires at least one member in 'members'.")
        self.members = [_make_member(m["type"], m.get("hp", {})) for m in members_spec]
        w = self.hp.get("weights", None)
        if w is None:
            self.weights = np.ones(len(self.members), dtype=float) / len(self.members)
        else:
            w = np.asarray(w, dtype=float)
            self.weights = w / (w.sum() if w.sum()!=0 else 1.0)

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        for m in self.members:
            m.fit(X_tr, y_tr)

    def predict_one(self, x_eval: np.ndarray) -> float:
        preds = np.array([m.predict_one(x_eval) for m in self.members], dtype=float)
        return float((self.weights * preds).sum())

    def get_name(self) -> str:
        return "ensemble_simple"
