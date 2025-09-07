from __future__ import annotations
from typing import List, Dict, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _ensure_df(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    raise TypeError("Expected pandas DataFrame or numpy array.")


# ------------------------------------------------------------
# IdentityTransformer (für Pipeline-Kompatibilität)
# ------------------------------------------------------------

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Gibt X unverändert zurück (behält DataFrame bei)."""
    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.columns_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return _ensure_df(X)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "columns_"):
            return np.array(self.columns_)
        if input_features is not None:
            return np.array(list(input_features))
        return np.array([f"x{i}" for i in range(getattr(self, "n_features_in_", 0))])


# ------------------------------------------------------------
# ColumnSelector
# ------------------------------------------------------------

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Wählt Spalten per manueller Liste und/oder Regex ein/aus."""

    def __init__(
        self,
        manual_features: Optional[List[str]] = None,
        include_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
    ):
        self.manual_features = manual_features
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex

    def fit(self, X, y=None):
        X = _ensure_df(X)
        cols = X.columns

        selected = cols
        if self.include_regex:
            selected = selected[selected.to_series().str.contains(self.include_regex, regex=True)]
        if self.exclude_regex:
            selected = selected[~selected.to_series().str.contains(self.exclude_regex, regex=True)]
        if self.manual_features:
            mf = [c for c in self.manual_features if c in cols]
            selected = pd.Index(mf) if mf else pd.Index([])

        self.selected_columns_ = pd.Index(selected)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        if len(self.selected_columns_) == 0:
            # leeres DF (kein Feature)
            return pd.DataFrame(index=X.index)
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.selected_columns_))


# ------------------------------------------------------------
# LagMaker  (SKLearn-klonbar)
# ------------------------------------------------------------

class LagMaker(BaseEstimator, TransformerMixin):
    """
    Erzeugt Lag-Features aus einem DataFrame.

    Parameter im __init__ werden NICHT verändert (wichtig für sklearn.clone).
    Alle Konvertierungen passieren in fit() und landen in self.lags_.
    """

    def __init__(self, lags: Iterable[int] = (1,), strategy: str = "value", ema_span: int = 3):
        self.lags = lags               # NICHT verändern!
        self.strategy = strategy       # "value" | "diff" | "mom" | "ema"
        self.ema_span = ema_span

    def fit(self, X, y=None):
        X = _ensure_df(X)

        # -> erst hier sauber konvertieren
        if isinstance(self.lags, Iterable) and not isinstance(self.lags, (str, bytes)):
            l = list(self.lags)
        else:
            l = [self.lags]

        l = [int(abs(int(k))) for k in l if int(k) > 0]
        if not l:
            raise ValueError("lags must contain at least one positive integer.")
        self.lags_ = tuple(sorted(set(l)))  # interne, saubere Repräsentation

        self.columns_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def _make_value(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for L in self.lags_:
            frames.append(X.shift(L).add_suffix(f"_lag{L}"))
        return pd.concat(frames, axis=1)

    def _make_diff(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for L in self.lags_:
            frames.append((X - X.shift(L)).add_suffix(f"_diff{L}"))
        return pd.concat(frames, axis=1)

    def _make_mom(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for L in self.lags_:
            mom = (X / X.shift(L) - 1.0) * 100.0
            frames.append(mom.add_suffix(f"_mom{L}"))
        return pd.concat(frames, axis=1)

    def _make_ema(self, X: pd.DataFrame) -> pd.DataFrame:
        ema = X.ewm(span=int(self.ema_span), adjust=False).mean().add_suffix(f"_ema{int(self.ema_span)}")
        frames = []
        for L in self.lags_:
            frames.append(ema.shift(L).add_suffix(f"_lag{L}"))
        return pd.concat(frames, axis=1)

    def transform(self, X):
        X = _ensure_df(X)
        if self.strategy == "value":
            out = self._make_value(X)
        elif self.strategy == "diff":
            out = self._make_diff(X)
        elif self.strategy in ("mom", "pct", "pct_change"):
            out = self._make_mom(X)
        elif self.strategy == "ema":
            out = self._make_ema(X)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'.")
        return out  # DataFrame zurückgeben (mit Spaltennamen)

    def get_feature_names_out(self, input_features=None):
        cols = self.columns_ if hasattr(self, "columns_") else (list(input_features) if input_features is not None else [])
        names = []
        if self.strategy == "value":
            for L in self.lags_:
                names += [f"{c}_lag{L}" for c in cols]
        elif self.strategy == "diff":
            for L in self.lags_:
                names += [f"{c}_diff{L}" for c in cols]
        elif self.strategy in ("mom", "pct", "pct_change"):
            for L in self.lags_:
                names += [f"{c}_mom{L}" for c in cols]
        elif self.strategy == "ema":
            for L in self.lags_:
                names += [f"{c}_ema{int(self.ema_span)}_lag{L}" for c in cols]
        return np.array(names)


# ------------------------------------------------------------
# ShockMonthDummyFromTarget
# ------------------------------------------------------------

class ShockMonthDummyFromTarget(BaseEstimator, TransformerMixin):
    """
    Erzeugt eine Dummy-Spalte für 'Schockmonate' der Zielvariable.
    - fit(): bestimmt Schockmonate NUR aus y_train (keine Leakage)
    - transform(): gibt Dummy (0/1) im Index von X zurück
    """

    def __init__(self, sigma: Optional[float] = None):
        self.sigma = sigma  # None -> inaktiv

    def fit(self, X, y=None):
        X = _ensure_df(X)
        if self.sigma is None or y is None:
            self.shock_months_ = pd.DatetimeIndex([])
            self.columns_ = list(X.columns)
            return self

        y = pd.Series(y, index=X.index[: len(y)]) if not isinstance(y, pd.Series) else y
        mu = float(y.mean())
        sd = float(y.std(ddof=0))
        thr = mu + float(self.sigma) * sd
        thr2 = mu - float(self.sigma) * sd
        mask = (y > thr) | (y < thr2)
        self.shock_months_ = pd.DatetimeIndex(y.index[mask])
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        if self.sigma is None or len(self.shock_months_) == 0:
            return pd.DataFrame(index=X.index)
        idx = pd.DatetimeIndex(X.index)
        dummy = idx.isin(self.shock_months_).astype(float)
        return pd.DataFrame({"shock_dummy": dummy}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(["shock_dummy"] if self.sigma is not None else [])


# ------------------------------------------------------------
# PerGroupTransformer
# ------------------------------------------------------------

class PerGroupTransformer(BaseEstimator, TransformerMixin):
    """
    Wendet einen Basis-Transformer getrennt je Feature-Gruppe an und konkateniert die Ergebnisse.
    """

    def __init__(self, base_transformer: TransformerMixin, groups: Optional[Dict[str, List[str]]] = None):
        self.base_transformer = base_transformer  # NICHT modifizieren
        self.groups = groups                      # NICHT modifizieren

    def fit(self, X, y=None):
        X = _ensure_df(X)
        if self.groups is None:
            grp = {"ALL": list(X.columns)}
        else:
            grp = {g: [c for c in cols if c in X.columns] for g, cols in self.groups.items() if len(cols) > 0}

        self.groups_ = grp
        self.transformers_ = {}
        self.feature_names_out_ = []

        for g, cols in self.groups_.items():
            if len(cols) == 0:
                continue
            Xt = X.loc[:, cols]
            m = clone(self.base_transformer)
            m.fit(Xt, y)
            self.transformers_[g] = m
            try:
                names = m.get_feature_names_out(cols)
                names = [f"{g}__{n}" for n in names]
            except Exception:
                names = [f"{g}__f{i}" for i in range(m.transform(Xt).shape[1])]
            self.feature_names_out_.extend(names)

        return self

    def transform(self, X):
        X = _ensure_df(X)
        outs = []
        for g, cols in self.groups_.items():
            if len(cols) == 0:
                continue
            Xt = X.loc[:, cols]
            m = self.transformers_[g]
            outs.append(_ensure_df(m.transform(Xt)))
        if not outs:
            return pd.DataFrame(index=X.index)
        out = pd.concat(outs, axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ------------------------------------------------------------
# FeatureFrameUnion  (horizontales Concatenate mehrerer Transformer)
# ------------------------------------------------------------

class FeatureFrameUnion(BaseEstimator, TransformerMixin):
    """
    Nimmt mehrere (name, transformer)-Paare und konkateniniert deren Ausgaben horizontal.
    Jeder Transformer erhält dasselbe (X, y).
    """

    def __init__(self, transformers: List[Tuple[str, TransformerMixin]]):
        self.transformers = transformers

    def fit(self, X, y=None):
        X = _ensure_df(X)
        self.fitted_ = []
        self.names_ = []
        self.feature_names_out_ = []
        for name, tr in self.transformers:
            m = clone(tr)
            m.fit(X, y)
            self.fitted_.append(m)
            self.names_.append(name)
            try:
                names = m.get_feature_names_out(getattr(X, "columns", None))
                names = [f"{name}__{n}" for n in names]
            except Exception:
                names = [f"{name}__f{i}" for i in range(_ensure_df(m.transform(X)).shape[1])]
            self.feature_names_out_.extend(names)
        return self

    def transform(self, X):
        X = _ensure_df(X)
        outs = [_ensure_df(m.transform(X)) for m in self.fitted_]
        if not outs:
            return pd.DataFrame(index=X.index)
        return pd.concat(outs, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
