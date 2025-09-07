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
# IdentityTransformer
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
# LagMaker  (klonbar + test-time Buffer aus Train)
# ------------------------------------------------------------

class LagMaker(BaseEstimator, TransformerMixin):
    """
    Erzeugt Lag-Features aus einem DataFrame.
    - Kein Imputing.
    - Beim transform() werden die letzten max(lags) Zeilen aus fit(X) als Buffer
      vorangestellt, damit im Test keine NaNs entstehen.
    """

    def __init__(self, lags: Iterable[int] = (1,), strategy: str = "value", ema_span: int = 3):
        self.lags = lags               # unverändert lassen (klonbar)
        self.strategy = strategy       # "value" | "diff" | "mom" | "ema"
        self.ema_span = ema_span

    def fit(self, X, y=None):
        X = _ensure_df(X)

        # lags sanitisieren
        if isinstance(self.lags, Iterable) and not isinstance(self.lags, (str, bytes)):
            l = list(self.lags)
        else:
            l = [self.lags]
        l = [int(abs(int(k))) for k in l if int(k) > 0]
        if not l:
            raise ValueError("lags must contain at least one positive integer.")
        self.lags_ = tuple(sorted(set(l)))
        self.max_lag_ = max(self.lags_)

        self.columns_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        # Buffer: letzte max_lag Zeilen aus Trainings-X
        self._buffer_ = X.tail(self.max_lag_).copy()
        return self

    # ---- interne Builders

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

    # ----

    def _build_features(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return out

    def transform(self, X):
        X = _ensure_df(X)
        if not hasattr(self, "_buffer_"):
            # falls direkt transform() ohne fit() aufgerufen wurde
            return self._build_features(X)

        # Train-Tail voranstellen, Features bauen, dann nur den Teil für X zurückgeben
        cat = pd.concat([self._buffer_, X], axis=0)
        feats = self._build_features(cat)

        # die letzten len(X) Zeilen (entsprechen X-Index) extrahieren
        out = feats.tail(len(X))
        # durch den Buffer sollten hier i.d.R. KEINE NaNs mehr sein
        return out

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
            span = int(self.ema_span)
            for L in self.lags_:
                names += [f"{c}_ema{span}_lag{L}" for c in cols]
        return np.array(names)


# ------------------------------------------------------------
# ShockMonthDummyFromTarget
# ------------------------------------------------------------

class ShockMonthDummyFromTarget(BaseEstimator, TransformerMixin):
    """
    Erzeugt eine Dummy-Spalte für 'Schockmonate' der Zielvariable (nur aus y_train).
    Kein Imputing.
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
# PerGroupTransformer  (präfixfreundlich)
# ------------------------------------------------------------

class PerGroupTransformer(BaseEstimator, TransformerMixin):
    """
    Wendet einen Basis-Transformer getrennt je Feature-Gruppe an und konkateniert die Ergebnisse.
    'groups' darf entweder exakte Spaltennamen ODER Präfixe enthalten.
    """

    def __init__(self, base_transformer: TransformerMixin, groups: Optional[Dict[str, List[str]]] = None):
        self.base_transformer = base_transformer  # NICHT modifizieren
        self.groups = groups                      # NICHT modifizieren

    def fit(self, X, y=None):
        X = _ensure_df(X)

        def expand_items(items, cols):
            items = list(items)
            exact = [c for c in items if c in cols]
            if not exact:
                expanded = [c for c in cols if any(c.startswith(p) for p in items)]
            else:
                expanded = set(exact)
                expanded.update(c for c in cols if any(c.startswith(p) for p in items))
                expanded = list(expanded)
            return list(dict.fromkeys(expanded))  # stable unique

        if self.groups is None:
            grp = {"ALL": list(X.columns)}
        else:
            grp = {}
            for g, items in self.groups.items():
                expanded = expand_items(items, list(X.columns))
                if expanded:
                    grp[g] = expanded

        if not grp:
            grp = {"ALL": list(X.columns)}

        self.groups_ = grp
        self.transformers_ = {}
        self.feature_names_out_ = []

        for g, cols in self.groups_.items():
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
