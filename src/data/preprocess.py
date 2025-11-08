import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Projekt-Root automatisch bestimmen (2 Ebenen hoch von dieser Datei)
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"

def load_features(path=DATA_PROCESSED / "panel_with_ip.csv", diff_type="pct"):
    """
    Lädt panel_with_ip.csv und baut Feature-Matrix (X) + Target (y).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    panel = pd.read_csv(path, parse_dates=["date"])

    panel_wide = panel.pivot_table(
        index="date", columns="indicator", values="value", aggfunc="last"
    )

    y = panel_wide["IP_change"].dropna()
    X = panel_wide.drop(columns=["IP_change"])

    if diff_type == "pct":
        X = X.pct_change() * 100
    elif diff_type == "diff":
        X = X.diff()

    X = X.ffill().fillna(0)
    X = X.loc[y.index]
    y = y.loc[X.index]

    return X, y


def expanding_window_splits(X, y, min_train_size=60, horizon=1, scale=True):
    scaler = StandardScaler() if scale else None

    for t in range(min_train_size, len(X) - horizon):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test, y_test   = X.iloc[t:t+horizon], y.iloc[t:t+horizon]

        if scaler:
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index, columns=X_train.columns
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index, columns=X_test.columns
            )

        yield X_train, y_train, X_test, y_test