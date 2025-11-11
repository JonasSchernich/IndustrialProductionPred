# src/models/LGBM.py
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    # sklearn-kompatibles API von LightGBM
    from lightgbm import LGBMRegressor
except Exception as e:
    raise ImportError("LightGBM (lightgbm) ist nicht installiert oder nicht importierbar.") from e


class ForecastModel:
    """
    LightGBM-Wrapper im ET/EN-Stil:
    - .fit(X, y, sample_weight=None)
    - .predict(X)
    - .predict_one(x_row)
    - .get_feature_importances()

    Wichtige HPs (Thesis ↔ LightGBM sklearn-API):
      - n_estimators             -> n_estimators
      - learning_rate            -> learning_rate
      - num_leaves               -> num_leaves
      - max_depth                -> max_depth
      - min_child_samples        -> min_child_samples   (Thesis: min_data_in_leaf)
      - subsample                -> subsample           (Thesis: bagging_fraction)
      - colsample_bytree         -> colsample_bytree    (Alias: feature_fraction)
      - reg_alpha                -> reg_alpha           (L1)
      - reg_lambda               -> reg_lambda          (L2)
      - min_split_gain           -> min_split_gain
      - min_child_weight         -> min_child_weight    (Thesis: min_sum_hessian_in_leaf)
      - max_bin                  -> max_bin
      - bagging_freq             -> subsample_freq
      - seed                     -> random_state
      - n_jobs                   -> n_jobs
      - importance_type          -> {'split','gain'} (für get_feature_importances)

    Optional (leakage-sicheres Early Stopping):
      - early_stopping_rounds (int)     -> aktiviert Early Stopping
      - val_tail (int)                  -> Größe des Validierungstails (z.B. 24). Wenn gesetzt,
                                           wird innerhalb .fit das letzte 'val_tail' der Train-Daten
                                           als eval_set genutzt (kein Look-ahead).

    Hinweise:
      * Upstream sollten Features train-only standardisiert sein (Z-Score), analog EN.
      * NaN/Inf werden als Sicherheitsnetz auf 0 gesetzt (analog ET/EN).
      * Für Vorhersagen wird automatisch best_iteration_ genutzt, sofern vorhanden.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._reg: Optional[LGBMRegressor] = None
        self._backend_name: str = "lightgbm"
        self._feature_names: Optional[List[str]] = None
        self._importances_cache: Optional[np.ndarray] = None
        self._importance_type: str = str(self.params.get("importance_type", "gain")).lower()

        # --- Hyperparameter mit robusten Defaults ---
        self._n_estimators: int      = int(self.params.get("n_estimators", 1000))
        self._learning_rate: float   = float(self.params.get("learning_rate", 0.05))
        self._num_leaves: int        = int(self.params.get("num_leaves", 31))
        self._max_depth: int         = int(self.params.get("max_depth", -1))  # -1 = unbegrenzt
        self._min_child_samples: int = int(self.params.get("min_child_samples", 20))  # = min_data_in_leaf
        self._subsample: float       = float(self.params.get("subsample", 1.0))       # bagging_fraction
        # LightGBM akzeptiert 'colsample_bytree'; 'feature_fraction' ist Alias in native-API
        self._colsample_bytree: float = float(self.params.get("colsample_bytree", self.params.get("feature_fraction", 1.0)))
        self._reg_alpha: float       = float(self.params.get("reg_alpha", 0.0))
        self._reg_lambda: float      = float(self.params.get("reg_lambda", 0.0))
        self._min_split_gain: float  = float(self.params.get("min_split_gain", 0.0))
        self._min_child_weight: float= float(self.params.get("min_child_weight", 1e-3))  # min_sum_hessian_in_leaf
        self._max_bin: int           = int(self.params.get("max_bin", 255))
        self._subsample_freq: int    = int(self.params.get("bagging_freq", self.params.get("subsample_freq", 0)))
        self._seed: int              = int(self.params.get("seed", 42))
        self._n_jobs: int            = int(self.params.get("n_jobs", 1))  # nested parallelism vermeiden
        self._verbosity: int         = int(self.params.get("verbosity", -1))  # -1: leise

        # Early-Stopping (optional, leakage-safe durch Tail-Split in fit)
        self._early_stopping_rounds: Optional[int] = None
        if "early_stopping_rounds" in self.params:
            esr = self.params.get("early_stopping_rounds")
            self._early_stopping_rounds = int(esr) if esr is not None else None
        self._val_tail: Optional[int] = None
        if "val_tail" in self.params:
            vt = self.params.get("val_tail")
            self._val_tail = int(vt) if vt is not None else None

        # LGBMRegressor-Instanz vorbereiten
        self._reg = LGBMRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            num_leaves=self._num_leaves,
            max_depth=self._max_depth if self._max_depth is not None else -1,
            min_child_samples=self._min_child_samples,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            min_split_gain=self._min_split_gain,
            min_child_weight=self._min_child_weight,
            max_bin=self._max_bin,
            subsample_freq=self._subsample_freq,
            random_state=self._seed,
            n_jobs=self._n_jobs,
            verbosity=self._verbosity,
        )

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _split_train_val_tail(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Optionaler Split in (Core, Tail) für Early Stopping.
        Gibt ((X_core, y_core, w_core), (X_val, y_val)) zurück; Tail kann None sein.
        """
        if (self._early_stopping_rounds is None) or (self._val_tail is None) or (self._val_tail <= 0):
            return (X, y, sample_weight), None

        n = X.shape[0]
        vt = min(self._val_tail, max(1, n // 5))  # pragmatische Obergrenze
        if n <= vt + 5:
            # Zu kurz, erspare Early Stopping
            return (X, y, sample_weight), None

        split = n - vt
        X_core, y_core = X[:split], y[:split]
        X_val,  y_val  = X[split:], y[split:]
        w_core = None if sample_weight is None else sample_weight[:split]
        return (X_core, y_core, w_core), (X_val, y_val)

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Feature-Namen erfassen
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        elif hasattr(X, "columns"):
            self._feature_names = list(X.columns)
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self._feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

        X_np = self._clean(X_np)
        y_np = np.asarray(y, dtype=float).ravel()

        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")

        # Optional: leakage-sicheres Early Stopping über Tail-Split
        (X_core, y_core, w_core), val_pair = self._split_train_val_tail(X_np, y_np, sw)

        if val_pair is not None and self._early_stopping_rounds:
            X_val, y_val = val_pair
            self._reg.fit(
                X_core, y_core,
                sample_weight=w_core,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                early_stopping_rounds=self._early_stopping_rounds,
                verbose=False,
            )
        else:
            self._reg.fit(X_np, y_np, sample_weight=sw)

        # Importances cachen (je nach importance_type)
        imp_type = "gain" if self._importance_type not in {"split", "gain"} else self._importance_type
        try:
            self._importances_cache = self._reg.feature_importances_  # LightGBM liefert standardmäßig Split-Zählungen
            if imp_type == "gain" and hasattr(self._reg, "booster_"):
                # Gain-Importances über Booster abrufen (robust gegen Versionen)
                booster = self._reg.booster_
                imp_gain = booster.feature_importance(importance_type="gain")
                if imp_gain is not None and np.sum(imp_gain) > 0:
                    self._importances_cache = imp_gain
        except Exception:
            self._importances_cache = None

        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        X_np = self._clean(X_np)

        # Nutze best_iteration, falls durch Early Stopping vorhanden
        num_it = getattr(self._reg, "best_iteration_", None)
        if num_it is not None and num_it > 0:
            return self._reg.predict(X_np, num_iteration=num_it)
        return self._reg.predict(X_np)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        if self._importances_cache is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances_cache))]
        return dict(zip(names, self._importances_cache.tolist()))
