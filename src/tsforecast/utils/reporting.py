from __future__ import annotations
from typing import Dict, List
import os
import pandas as pd

def append_summary_row(path: str, row: Dict):
    df = pd.DataFrame([row])
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)

def append_tuning_rows(path: str, rows: List[Dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)

def append_predictions_rows(path: str, rows: List[Dict]):
    """
    rows: [{"time": <pd.Timestamp|str>, "model": str, "y_true": float, "y_hat": float}, ...]
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)
