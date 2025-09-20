
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_metric(name: str):
    name = name.lower()
    if name == "mae":
        return lambda yt, yp: mean_absolute_error(yt, yp)
    if name == "rmse":
        return lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
    if name == "mse":
        return lambda yt, yp: float(mean_squared_error(yt, yp))
    raise ValueError(f"Unknown metric {name}")
