# src/utils/time.py
from __future__ import annotations
from typing import Iterable
import pandas as pd


def to_month_end(idx: pd.DatetimeIndex | Iterable) -> pd.DatetimeIndex:
    """
    Stellt sicher, dass alle Zeitstempel auf Monatsende liegen (freq='M'-kompatibel).
    """
    idx = pd.to_datetime(idx)
    return idx.to_series().dt.to_period("M").dt.to_timestamp("M").to_numpy()


def ensure_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert den Index eines DataFrames auf Monatsende (falls nicht schon so).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.copy()
    df.index = pd.DatetimeIndex(to_month_end(df.index), name=df.index.name)
    return df


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """Addiert n Monate zu einem Timestamp (nutzt Period arithmetics)."""
    return (ts.to_period("M") + n).to_timestamp("M")
