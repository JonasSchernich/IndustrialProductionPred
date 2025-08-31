import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BaseModel:
    def fit(self, y):
        self.y_train = y.dropna()
        return self

    def predict(self, horizon=1):
        raise NotImplementedError

    def score(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # <-- RMSE manuell
        return {"MAE": mae, "MSE": mse, "RMSE": rmse}


class MeanModel(BaseModel):
    def fit(self, y: pd.Series):
        super().fit(y)
        self.mean_ = self.y_train.mean()
        return self

    def predict(self, horizon=1):
        return np.repeat(self.mean_, horizon)


class RandomWalkModel(BaseModel):
    def fit(self, y: pd.Series):
        super().fit(y)
        return self

    def predict(self, horizon=1):
        last_value = self.y_train.iloc[-1]
        return np.repeat(last_value, horizon)


class AR1Model(BaseModel):
    def fit(self, y: pd.Series):
        super().fit(y)
        y_lag = self.y_train.shift(1).dropna()
        y_curr = self.y_train.iloc[1:]
        phi = np.linalg.lstsq(
            y_lag.values.reshape(-1, 1),
            y_curr.values,
            rcond=None
        )[0][0]
        self.phi_ = phi
        self.alpha_ = y_curr.mean() - phi * y_lag.mean()
        return self

    def predict(self, horizon=1):
        preds = []
        last_val = self.y_train.iloc[-1]
        for _ in range(horizon):
            next_val = self.alpha_ + self.phi_ * last_val
            preds.append(next_val)
            last_val = next_val
        return np.array(preds)
