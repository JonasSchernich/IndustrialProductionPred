import pandas as pd

X = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
y = pd.read_csv("data/processed/target.csv",   index_col=0, parse_dates=True).iloc[:,0]

print("X rows:", len(X), "y rows:", len(y))
print("X idx dtype:", X.index.dtype, "y idx dtype:", y.index.dtype)
print("X range:", X.index.min(), "→", X.index.max())
print("y range:", y.index.min(), "→", y.index.max())

# Normalisiere Index auf Monatsanfang (falls noch nicht perfekt)
X.index = pd.to_datetime(X.index).to_period("M").to_timestamp("MS")
y.index = pd.to_datetime(y.index).to_period("M").to_timestamp("MS")
X = X[~X.index.duplicated()]
y = y[~y.index.duplicated()]

common = X.index.intersection(y.index)
print("COMMON rows:", len(common), "range:", common.min(), "→", common.max())

# Wieviele Folds wären möglich?
initial_window, horizon = 60, 1
print("min required >", initial_window + horizon, "; n=", len(common))
