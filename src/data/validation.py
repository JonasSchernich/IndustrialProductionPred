# src/data/validation.py
from __future__ import annotations
import argparse, pathlib
import pandas as pd

def validate_features(path: str | pathlib.Path) -> pd.DataFrame:
    X = pd.read_csv(path, index_col=0, parse_dates=True)
    report = []
    report.append(("n_rows", len(X)))
    report.append(("n_cols", X.shape[1]))
    report.append(("index_monotonic", bool(X.index.is_monotonic_increasing)))
    report.append(("index_unique", bool(X.index.is_unique)))
    report.append(("nans_total", int(X.isna().sum().sum())))
    report.append(("cols_all_na", int((X.isna().all(axis=0)).sum())))
    miss_cols = (X.isna().mean()*100.0).sort_values(ascending=False)
    top5 = ", ".join([f"{c}:{miss_cols[c]:.1f}%" for c in miss_cols.index[:5]])
    report.append(("top5_missing_cols", top5))
    return pd.DataFrame(report, columns=["check","value"])

def validate_target(path: str | pathlib.Path) -> pd.DataFrame:
    y = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:,0]
    report = []
    report.append(("n_rows", len(y)))
    report.append(("index_monotonic", bool(y.index.is_monotonic_increasing)))
    report.append(("index_unique", bool(y.index.is_unique)))
    report.append(("nans_total", int(y.isna().sum())))
    return pd.DataFrame(report, columns=["check","value"])

def validate_alignment(features_path: str | pathlib.Path, target_path: str | pathlib.Path) -> pd.DataFrame:
    X = pd.read_csv(features_path, index_col=0, parse_dates=True)
    y = pd.read_csv(target_path, index_col=0, parse_dates=True).iloc[:,0]
    common = X.index.intersection(y.index)
    report = []
    report.append(("X_rows", len(X)))
    report.append(("y_rows", len(y)))
    report.append(("common_rows", len(common)))
    report.append(("coverage_y_in_X_%", round(100*len(common)/len(y), 1) if len(y)>0 else 0))
    return pd.DataFrame(report, columns=["check","value"])

def main():
    ap = argparse.ArgumentParser(description="Validate processed features/target files.")
    ap.add_argument("--features", type=str, default="data/processed/features.csv")
    ap.add_argument("--target", type=str, default="data/processed/target.csv")
    args = ap.parse_args()

    f_rep = validate_features(args.features)
    t_rep = validate_target(args.target)
    a_rep = validate_alignment(args.features, args.target)

    print("[Features]")
    print(f_rep.to_string(index=False))
    print("\n[Target]")
    print(t_rep.to_string(index=False))
    print("\n[Alignment]")
    print(a_rep.to_string(index=False))

if __name__ == "__main__":
    main()
