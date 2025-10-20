
from __future__ import annotations
from typing import Iterable, Iterator, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import os

from .config import PROCESSED, outputs_for_model, GlobalConfig

def _validate_index(idx: pd.DatetimeIndex) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be strictly increasing.")
    # monthly frequency is recommended but not enforced strictly here

def load_target(col: str = "IP_change") -> pd.Series:
    """
    Lädt target.csv und gibt standardmäßig die Spalte 'IP_change' zurück.
    Minimaler Patch gegenüber der alten Version (nur Spaltenwahl geändert).
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

def load_tsfresh() -> pd.DataFrame:
    path = PROCESSED / "tsfresh_slim.parquet"   # <- passt zu deinem Export
    df = pd.read_parquet(path)
    return df

def load_chronos() -> pd.DataFrame:
    path = PROCESSED / "chronos_bolt.parquet"   # <- an deinen Dateinamen anpassen
    df = pd.read_parquet(path)
    return df


def stageA_blocks(cfg: GlobalConfig, T: int) -> Iterator[Tuple[int,int,int,int]]:
    """
    Yields (train_end, oos_start, oos_end, block_id) with inclusive bounds for OOS.
    Training starts at 1 .. train_end (W0_A), OOS is per defined block.
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
