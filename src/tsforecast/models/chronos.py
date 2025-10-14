# tsforecast/models/chronos.py
from __future__ import annotations
import numpy as np, torch
from chronos import ChronosPipeline

class ChronosRegressor:
    def __init__(self,
                 model_id: str = "amazon/chronos-t5-tiny",
                 use_gpu: bool = False,
                 torch_dtype: str = "float32",
                 num_samples: int = 20):
        self.model_id = model_id
        self.device_map = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.dtype = getattr(torch, torch_dtype, torch.float32)
        self.num_samples = int(num_samples)
        self.pipe = None
        self.hist = None

    def _ensure_loaded(self):
        if self.pipe is None:
            self.pipe = ChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device_map, torch_dtype=self.dtype
            )

    def fit(self, X, y):
        self._ensure_loaded()
        self.hist = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, X):
        self._ensure_loaded()
        if self.hist is None or self.hist.size == 0:
            return np.asarray([np.nan], dtype=float)
        ctx = torch.tensor(self.hist, dtype=torch.float32)
        try:
            samples = self.pipe.predict(context=ctx, prediction_length=1, num_samples=self.num_samples)
            v = float(samples.mean().item())
        except TypeError:
            samples = self.pipe.predict(ctx, horizon=1, num_samples=self.num_samples)
            v = float(samples.mean().item())
        return np.asarray([v], dtype=float)
