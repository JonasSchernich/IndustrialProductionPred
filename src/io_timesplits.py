# src/io_timesplits.py

from __future__ import annotations
from typing import Iterable, Iterator, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- KORREKTUR: Relative Importe (wie gewünscht) ---
from .config import PROCESSED, outputs_for_model, GlobalConfig


def _validate_index(idx: pd.DatetimeIndex) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be strictly increasing.")


def load_target(col: str = "IP_change") -> pd.Series:
    """
    Lädt target.csv und gibt standardmäßig die Spalte 'IP_change' zurück.
    """
    path = PROCESSED / "target.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if col not in df.columns:
        raise ValueError(
            f"Spalte '{col}' nicht in target.csv gefunden. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )
    s = df[col].astype(float).dropna().sort_index()
    _validate_index(s.index)
    return s


def load_ifo_features() -> pd.DataFrame:
    path = PROCESSED / "cleaned_features.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    _validate_index(df.index)
    return df


# --- NEU: Parquet-Ladefunktion (behebt Timestamp-Anzeigeproblem) ---

def _load_parquet_with_datetime_index(
        filename: str,
        date_col: str = "date"
) -> pd.DataFrame:
    """
    Lädt eine Parquet-Datei, die einen Datums-Index als Spalte gespeichert hat.
    Setzt diesen Index als DatetimeIndex.
    """
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    df = pd.read_parquet(path)

    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            _validate_index(df.index)
            return df
        raise ValueError(
            f"Datumsspalte '{date_col}' nicht in {filename} gefunden "
            f"und der Index ist kein DatetimeIndex."
        )

    # Konvertiere die Spalte (kann int/Nanosekunden sein) in Datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    _validate_index(df.index)
    return df


def load_tsfresh() -> pd.DataFrame:
    """Lädt TSFresh und stellt DatetimeIndex sicher."""
    return _load_parquet_with_datetime_index("tsfresh_slim.parquet", date_col="date")


def load_chronos() -> pd.DataFrame:
    """Lädt Chronos und stellt DatetimeIndex sicher."""
    return _load_parquet_with_datetime_index("chronos_bolt.parquet", date_col="date")


def load_ar() -> pd.DataFrame:
    """NEU: Lädt das AR.parquet-Feature und stellt DatetimeIndex sicher."""
    return _load_parquet_with_datetime_index("AR.parquet", date_col="date")


# --- ENDE PARQUET-Ladefunktion ---


def stageA_blocks(cfg: GlobalConfig, T: int) -> Iterator[Tuple[int, int, int, int]]:
    """
    Yields (train_end, oos_start, oos_end, block_id) with inclusive bounds for OOS.
    """
    blocks = list(cfg.BLOCKS_A)
    for i, (oos_start, oos_end) in enumerate(blocks, start=1):
        train_end = cfg.W0_A if i == 1 else (oos_start - 1)
        yield (train_end, oos_start, oos_end, i)


def stageB_months(cfg: GlobalConfig, T: int) -> Iterator[int]:
    """Yields origins t = cfg.W0_B .. T-1 (predict t+1)."""
    for t in range(cfg.W0_B, T):
        yield t


def ensure_outputs(model_name: str) -> Dict[str, Path]:
    return outputs_for_model(model_name)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)