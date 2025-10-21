# src/io_timesplits.py

from __future__ import annotations
from typing import Iterable, Iterator, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import os, re

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


def _clean_column_name(col_name: str) -> str:
    """Ensures column names are valid identifiers (strings, no weird chars)."""
    name = str(col_name)
    # Replace invalid characters (anything not letter, number, or _) with underscore
    name = re.sub(r'[^a-zA-Z0-9_.]', '_', name)
    # Prepend 'f_' if the name starts with a digit or period
    if re.match(r'^[0-9.]', name):
        name = 'f_' + name
    # Ensure name is not empty
    if not name:
        name = 'unnamed_feature'
    return name


def load_ifo_features() -> pd.DataFrame:
    path = PROCESSED / "cleaned_features.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    _validate_index(df.index)

    # --- NEW: Clean column names ---
    original_columns = df.columns.tolist()
    cleaned_columns = [_clean_column_name(col) for col in original_columns]

    # Check if renaming actually happened
    if original_columns != cleaned_columns:
        print("INFO in load_ifo_features: Renaming columns to ensure validity.")
        # Create a mapping for clarity (optional logging)
        # name_mapping = dict(zip(original_columns, cleaned_columns))
        # print("Name mapping:", name_mapping)
        df.columns = cleaned_columns
    # --- END NEW ---

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