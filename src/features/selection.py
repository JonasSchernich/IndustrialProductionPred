from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


def _as_df(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    raise TypeError("Expected pandas DataFrame or numpy array")


# ---------------------------------------------------------------------
# Varianz-Filter (spaltenweise)
# ---------------------------------------------------------------------
class VarianceThresholdColumns(BaseEstimator, TransformerMixin):
    """Entfernt Spalten mit Varianz <= threshold (NaNs werden ignoriert)."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        X = _as_df(X)
        # Varianz mit NaN-Ignore
        var = X.var(axis=0, skipna=True, ddof=0)
        self.keep_cols_ = var[var > self.threshold].index.to_list()
        return self

    def transform(self, X):
        X = _as_df(X)
        if len(self.keep_cols_) == 0:
            return pd.DataFrame(index=X.index)
        return X.loc[:, self.keep_cols_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


# ---------------------------------------------------------------------
# Korrelation Feature ↔ Target (top-k, min_abs)
# ---------------------------------------------------------------------
class CorrWithTargetSelector(BaseEstimator, TransformerMixin):
    """
    Wählt die top_k Spalten nach |corr(feature, y)| (Pearson),
    zusätzlich Filter: |corr| >= min_abs.
    NaNs werden paarweise ignoriert (pairwise complete).
    """

    def __init__(self, top_k: int = 100, min_abs: float = 0.0):
        self.top_k = int(top_k)
        self.min_abs = float(min_abs)

    def fit(self, X, y=None):
        X = _as_df(X)
        if y is None:
            raise ValueError("CorrWithTargetSelector requires y in fit()")

        y = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index[: len(y)])
        # pairwise corr (pandas behandelt NaNs paarweise)
        corr = X.corrwith(y)
        corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
        corr_abs = corr.abs()

        # min_abs-Filter
        filt = corr_abs >= self.min_abs
        corr_abs = corr_abs[filt]

        # top-k
        top_cols = corr_abs.sort_values(ascending=False).head(self.top_k).index.to_list()
        self.keep_cols_ = top_cols
        return self

    def transform(self, X):
        X = _as_df(X)
        if len(self.keep_cols_) == 0:
            return pd.DataFrame(index=X.index)
        keep = [c for c in self.keep_cols_ if c in X.columns]
        return X.loc[:, keep]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


# ---------------------------------------------------------------------
# Inter-Feature-Korrelation (redundante Spalten entfernen)
# ---------------------------------------------------------------------
class InterFeatureCorrSelector(BaseEstimator, TransformerMixin):
    """
    Entfernt Spalten, die untereinander zu hoch korrelieren (|rho| >= threshold).
    Strategie:
      - method="first": behalte die zuerst gesehene Spalte, entferne nachfolgende hochkorrelierte
      - method="var"  : sortiere Spalten nach Varianz (absteigend) und behalte jeweils die mit höherer Varianz
    """

    def __init__(self, threshold: float = 0.95, method: str = "var"):
        self.threshold = float(threshold)
        self.method = str(method)

    def fit(self, X, y=None):
        X = _as_df(X)
        # für Stabilität: nur numerische Spalten
        Xn = X.select_dtypes(include=[np.number])

        # Zeilen mit NaN entfernen für Corr-Berechnung
        Xn_drop = Xn.dropna(axis=0, how="any")
        if Xn_drop.shape[0] < 2 or Xn_drop.shape[1] == 0:
            self.keep_cols_ = list(Xn.columns)
            return self

        corr = Xn_drop.corr().abs()

        if self.method == "var":
            order = Xn_drop.var(axis=0, ddof=0).sort_values(ascending=False).index.to_list()
        else:  # "first"
            order = list(Xn_drop.columns)

        keep: List[str] = []
        dropped: set = set()

        for c in order:
            if c in dropped:
                continue
            keep.append(c)
            # alle stark korrelierten Spalten zu c entfernen
            high = corr.index[(corr[c] >= self.threshold) & (corr.index != c)].to_list()
            dropped.update(high)

        self.keep_cols_ = keep
        return self

    def transform(self, X):
        X = _as_df(X)
        keep = [c for c in self.keep_cols_ if c in X.columns]
        if len(keep) == 0:
            return pd.DataFrame(index=X.index)
        return X.loc[:, keep]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


# ---------------------------------------------------------------------
# PCA je Gruppe (optionale Dimensionsreduktion)
# ---------------------------------------------------------------------
class PCAByGroup(BaseEstimator, TransformerMixin):
    """
    Führt pro Gruppe (Mapping: group -> [cols]) eine PCA durch.
    NaNs werden durch Spaltenmittelwerte (train) ersetzt.
    """

    def __init__(self, groups: Optional[Dict[str, List[str]]] = None, n_components: int = 1, random_state: int = 42):
        self.groups = groups   # im __init__ nicht verändern (sklearn.clone)
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, X, y=None):
        X = _as_df(X)
        if self.groups is None:
            grp = {"ALL": list(X.columns)}
        else:
            grp = {g: [c for c in cols if c in X.columns] for g, cols in self.groups.items() if len(cols) > 0}

        self.groups_ = grp
        self.models_: Dict[str, PCA] = {}
        self.means_: Dict[str, pd.Series] = {}
        self.out_columns_: List[str] = []

        for g, cols in self.groups_.items():
            if len(cols) == 0:
                continue
            G = X.loc[:, cols]
            means = G.mean(axis=0, skipna=True)
            self.means_[g] = means

            G_filled = G.fillna(means)
            # PCA fit
            k = min(self.n_components, G_filled.shape[1])
            if k < 1:
                continue
            pca = PCA(n_components=k, random_state=self.random_state)
            pca.fit(G_filled.values)
            self.models_[g] = pca
            self.out_columns_.extend([f"{g}__PC{i+1}" for i in range(k)])

        return self

    def transform(self, X):
        X = _as_df(X)
        outs = []
        for g, cols in self.groups_.items():
            if g not in self.models_:
                continue
            G = X.loc[:, [c for c in cols if c in X.columns]]
            means = self.means_[g]
            G = G.reindex(columns=means.index)  # gleiche Reihenfolge
            G_filled = G.fillna(means)
            Z = self.models_[g].transform(G_filled.values)
            cols_out = [f"{g}__PC{i+1}" for i in range(Z.shape[1])]
            outs.append(pd.DataFrame(Z, index=X.index, columns=cols_out))
        if not outs:
            return pd.DataFrame(index=X.index)
        return pd.concat(outs, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.out_columns_)
