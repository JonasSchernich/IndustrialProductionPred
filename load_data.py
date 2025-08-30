# load_data.py
import pandas as pd, re, glob, pathlib, numpy as np

BASE_DIR   = pathlib.Path(__file__).parent
BASE_DIR   = BASE_DIR / "data"
DATA_DIR   = BASE_DIR / "raw_feature_data"
TARGET_DIR = BASE_DIR / "raw_target_data"
PROCESSED_DIR   = BASE_DIR / "loaded_data"
OUT_CSV_PANEL_LONG        = PROCESSED_DIR / "ifo_panel.csv"
OUT_CSV_PANEL_LONG_WITH_IP= PROCESSED_DIR / "panel_with_ip.csv"
OUT_PARQUET_X             = PROCESSED_DIR / "X.parquet"
OUT_CSV_Y                 = PROCESSED_DIR / "y.csv"

PAT = re.compile(r'^(?P<indicator>.+?)\s*\((?P<info>[A-Z])\)\s*(?P<branch>.+?)(?:\s+BD\s+SBR)?$')
DOT_DDMMYY = re.compile(r'^\s*\d{2}\.\d{2}\.(\d{2}|\d{4})\s*$')

def parse_title(s):
    if not isinstance(s, str):
        return {"indicator": None, "additional_info": None, "branch": None}
    s = re.sub(r'\s+', ' ', s).strip()
    m = PAT.match(s)
    if not m:
        return {"indicator": None, "additional_info": None, "branch": s}
    d = m.groupdict()
    return {
        "indicator": d["indicator"].strip(),
        "branch": d["branch"].strip(),
        "additional_info": d.get("info")
    }

def _to_month_start(ts):
    if pd.isna(ts): return pd.NaT
    ts = pd.Timestamp(ts)
    return pd.Timestamp(ts.year, ts.month, 1)

def parse_month(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)) and not np.isnan(x):
        # Excel-Seriennummern
        dt = pd.to_datetime(x, unit='d', origin='1899-12-30', errors='coerce')
        return _to_month_start(dt)
    s = str(x).strip()

    # dd.mm.yy / dd.mm.yyyy → dayfirst
    if DOT_DDMMYY.fullmatch(s):
        dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
        return _to_month_start(dt)

    # Normalisierungen
    s2 = s.replace('.', '/').replace('-', '/')

    # mm/YYYY
    if re.fullmatch(r'\d{2}/\d{4}', s2):
        dt = pd.to_datetime(s2, format='%m/%Y', errors='coerce')
        return _to_month_start(dt)
    # YYYY/mm
    if re.fullmatch(r'\d{4}/\d{2}', s2):
        y, m = s2.split('/')
        return pd.Timestamp(int(y), int(m), 1)

    # Fallback
    dt = pd.to_datetime(s2, errors='coerce', dayfirst=True)
    return _to_month_start(dt) if pd.notna(dt) else pd.NaT

def load_file(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=2, engine="openpyxl")
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = df["date"].map(parse_month)

    value_cols = [c for c in df.columns[1:] if not df[c].isna().all()]
    df = df[["date"] + value_cols]

    long = df.melt(id_vars="date", var_name="title_raw", value_name="value")
    parts = long["title_raw"].map(parse_title).apply(pd.Series)
    out = pd.concat([long, parts], axis=1)

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["file"]  = path.name

    out["branch"]    = out["branch"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["indicator"] = out["indicator"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return out[["date","branch","indicator","additional_info","value","title_raw","file"]]

def load_all_features_long() -> pd.DataFrame:
    files = sorted(map(pathlib.Path, glob.glob(str(DATA_DIR / "bdi1a_*.xlsx"))))
    if not files:
        raise FileNotFoundError(f"keine .xlsx unter {DATA_DIR}/bdi1a_*.xlsx")
    panel = pd.concat([load_file(f) for f in files], ignore_index=True)
    panel = panel.sort_values(["date","branch","indicator"]).reset_index(drop=True)
    return panel

def load_ip_target(xlsx_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="IP", engine="openpyxl")
    date_col = df.columns[0]
    # Spalte 'IP' robust finden
    ip_col = next((c for c in df.columns if str(c).strip().lower() == "ip"), df.columns[1])
    df = df[[date_col, ip_col]].rename(columns={date_col: "date", ip_col: "IP"})
    df["date"] = df["date"].map(parse_month)
    df["IP"] = pd.to_numeric(df["IP"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df

def make_series_name(row: pd.Series) -> str:
    # stabile, eindeutige Spaltennamen für die breite Matrix
    parts = [
        (row.get("indicator") or "").strip(),
        (row.get("branch") or "").strip(),
        (row.get("additional_info") or ""),
    ]
    name = "|".join(parts)
    name = re.sub(r"[^0-9A-Za-z|]+", "_", name).strip("_")
    name = name.replace("||", "|").strip("|")
    return name or "unknown"

def features_long_to_wide(panel_long: pd.DataFrame) -> pd.DataFrame:
    s = panel_long.apply(make_series_name, axis=1)
    tmp = panel_long.assign(series=s)
    wide = tmp.pivot_table(index="date", columns="series", values="value", aggfunc="last")
    wide = wide.sort_index()
    # Speicher schonen
    wide = wide.astype("float32")
    # Spaltennamen säubern
    safe_cols = [re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_") for c in wide.columns]
    wide.columns = safe_cols
    return wide

def add_ip_to_long(panel_long: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    ip_long = ip_df.assign(
        branch="Total",
        indicator="IP",
        additional_info=None,
        value=lambda d: d["IP"],
        title_raw="IP",
        file="IndustrialProd.xlsx"
    )[["date","branch","indicator","additional_info","value","title_raw","file"]]
    out = pd.concat([panel_long, ip_long], ignore_index=True)
    out = out.sort_values(["date","branch","indicator"]).reset_index(drop=True)
    return out

def build_X_y(panel_long: pd.DataFrame, ip_df: pd.DataFrame):
    X = features_long_to_wide(panel_long)
    y = ip_df.set_index("date").sort_index()["IP"].astype("float32")
    # nur Monate mit IP behalten
    data = X.join(y, how="inner")
    y = data.pop("IP")
    X = data
    return X, y

if __name__ == "__main__":
    # 1) Features (long)
    panel_long = load_all_features_long()
    panel_long.to_csv(OUT_CSV_PANEL_LONG, index=False, encoding="utf-8-sig")
    print(f"panel long: {len(panel_long):,} zeilen")

    # 2) Target IP
    ip_path = TARGET_DIR / "IndustrialProd.xlsx"
    ip_df = load_ip_target(ip_path)
    print(f"ip: {len(ip_df):,} monate")

    # 3) Optional: IP ans long-Panel für EDA
    panel_with_ip = add_ip_to_long(panel_long, ip_df)
    panel_with_ip.to_csv(OUT_CSV_PANEL_LONG_WITH_IP, index=False, encoding="utf-8-sig")
    print("panel+ip: ok")

    # 4) Modell-Matrix
    X, y = build_X_y(panel_long, ip_df)
    X.to_parquet(OUT_PARQUET_X, index=True)
    y.to_csv(OUT_CSV_Y, index=True, encoding="utf-8-sig")
    print(f"X: {X.shape}, y: {y.shape}")
