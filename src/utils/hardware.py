# src/utils/hardware.py
from __future__ import annotations
from typing import Optional
import os
import random
import numpy as np


def detect_device(prefer: str = "cuda") -> str:
    """
    Gibt 'cuda' zurück, wenn PyTorch verfügbar ist und CUDA hat – sonst 'cpu'.
    """
    if prefer != "cuda":
        return "cpu"
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # deterministische Optionen (langsamer)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def effective_n_jobs(n_jobs: Optional[int]) -> int:
    """
    Wandelt None/0/negatives in eine sinnvolle Anzahl Threads um.
    """
    if n_jobs is None or n_jobs == 0:
        return max(1, os.cpu_count() or 1)
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)  # sklearn-Konvention
    return n_jobs
