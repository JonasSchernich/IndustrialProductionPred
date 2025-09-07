# src/evaluation/splitters.py
from __future__ import annotations
from typing import Iterator, Tuple, Optional
import numpy as np
from sklearn.model_selection import BaseCrossValidator


class ExpandingWindowSplit(BaseCrossValidator):
    """
    Expanding-Window Cross-Validator für Zeitreihen.

    initial_window : Größe des ersten Trainingsfensters (int)
    step           : wie viele Zeitpunkte pro Fold vorwärts (int, default 1)
    horizon        : Testhorizont je Fold (int, default 1)
    max_train_size : optional, cap der Trainingslänge (None = unbegrenzt)
    """
    def __init__(self, initial_window: int, step: int = 1, horizon: int = 1, max_train_size: Optional[int] = None):
        if initial_window <= 0:
            raise ValueError("initial_window must be > 0")
        if step <= 0:
            raise ValueError("step must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        self.initial_window = int(initial_window)
        self.step = int(step)
        self.horizon = int(horizon)
        self.max_train_size = max_train_size if max_train_size is None else int(max_train_size)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        t = self.initial_window
        while t + self.horizon <= n:
            train_start = 0 if self.max_train_size is None else max(0, t - self.max_train_size)
            train_idx = np.arange(train_start, t)
            test_idx = np.arange(t, min(t + self.horizon, n))
            yield train_idx, test_idx
            t += self.step

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            raise ValueError("X required to compute number of splits.")
        n = len(X)
        return max(0, (n - self.initial_window) // self.step)


class NestedExpandingWindowSplit:
    """
    Verschachtelter Expanding-Window-Split:
    - outer: liefert (train_outer, test_outer)
    - inner: für jedes outer-Fenster kleineres Expanding-CV zum Tuning

    Nutzung:
      outer = NestedExpandingWindowSplit( ... )
      for (tr_o, te_o), inner_cv in outer.split(X):
          # inner_cv ist ein ExpandingWindowSplit, das auf X[tr_o] angewandt wird
    """
    def __init__(
        self,
        outer_initial: int,
        outer_step: int = 1,
        outer_horizon: int = 1,
        inner_initial: Optional[int] = None,
        inner_step: int = 1,
        inner_horizon: int = 1,
        max_train_size: Optional[int] = None,
    ):
        self.outer_initial = int(outer_initial)
        self.outer_step = int(outer_step)
        self.outer_horizon = int(outer_horizon)
        self.inner_initial = int(inner_initial) if inner_initial is not None else None
        self.inner_step = int(inner_step)
        self.inner_horizon = int(inner_horizon)
        self.max_train_size = max_train_size if max_train_size is None else int(max_train_size)

    def split(self, X, y=None, groups=None):
        outer_cv = ExpandingWindowSplit(
            initial_window=self.outer_initial,
            step=self.outer_step,
            horizon=self.outer_horizon,
            max_train_size=self.max_train_size,
        )
        for tr_o, te_o in outer_cv.split(X):
            inner_init = self.inner_initial if self.inner_initial is not None else max(5, len(tr_o) // 3)
            inner_cv = ExpandingWindowSplit(
                initial_window=inner_init,
                step=self.inner_step,
                horizon=self.inner_horizon,
                max_train_size=self.max_train_size,
            )
            yield (tr_o, te_o), inner_cv
