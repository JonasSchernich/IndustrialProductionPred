# models/chronos.py
import numpy as np

class ChronosRegressor:
    def __init__(self, model_id="amazon/chronos-t5-tiny", device_map="auto", torch_dtype="bfloat16"):
        try:
            from chronos import ChronosPipeline
            import torch
        except Exception as e:
            raise ImportError("pip install chronos-forecasting torch") from e
        self._ChronosPipeline = ChronosPipeline
        import torch as _torch
        self._torch = _torch
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = getattr(_torch, torch_dtype)
        self.pipe = None
        self.hist = None

    def fit(self, X, y):
        if self.pipe is None:
            self.pipe = self._ChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device_map, torch_dtype=self.torch_dtype
            )
        self.hist = self._torch.tensor(np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        fc = self.pipe.predict(self.hist, horizon=1)
        return np.asarray([float(fc[0])])
