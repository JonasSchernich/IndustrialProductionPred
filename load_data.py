# load_data.py
import pandas as pd, re, glob, pathlib, numpy as np

BASE_DIR = pathlib.Path(__file__).parents[1] / "Code"
DATA_DIR = BASE_DIR / "data" / "raw" / "features"
TARGET_DIR = BASE_DIR / "data" / "raw" / "target"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUT_CSV_PANEL_LONG        = PROCESSED_DIR / "ifo_panel.csv"
OUT_CSV_PANEL_LONG_WITH_IP= PROCESSED_DIR / "panel_with_ip.csv"
OUT_TARGET_CSV            = PROCESSED_DIR / "target.csv"

PAT = re.compile(r'^(?P<indicator>.+?)\s*\((?P<info>[A-Z])\)\s*(?P<branch>.+?)(?:\s+BD\s+SBR)?$')
DOT_DDMMYY = re.compile(r'^\s*\d{2}\.\d{2}\.(\d{2}|\d{4})\s*$')

# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------

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
        dt = pd.to_datetime(x, unit='d', origin='1899-12-30', errors='coerce')
        return _to_month_start(dt)
    s = str(x).strip()
    if DOT_DDMMYY.fullmatch(s):
        dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
        return _to_month_start(dt)
    if re.fullmatch(r"\d{2}[./]\d{4}", s):
        dt = pd.to_datetime(s.replace(".", "/"), format="%m/%Y", errors="coerce")
        return _to_month_start(dt)
    if re.fullmatch(r"\d{4}[./]\d{2}", s):
        y, m = re.split(r"[./]", s)
        return pd.Timestamp(int(y), int(m), 1)
    if re.fullmatch(r"\d{4}", s):
        return pd.Timestamp(int(s), 1, 1)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return _to_month_start(dt) if pd.notna(dt) else pd.NaT

# --------------------------------------------------
# Features laden
# --------------------------------------------------

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

# --------------------------------------------------
# Target laden
# --------------------------------------------------

def load_ip_target(xlsx_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="IP", engine="openpyxl")
    df = df.rename(columns={
        df.columns[0]: "date",
        "IP": "IP",
        "MoM": "IP_change",
        "YoY": "IP_yoy"
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "IP", "IP_change", "IP_yoy"]]

# --------------------------------------------------
# Panel mit Target mergen
# --------------------------------------------------

def add_ip_to_long(panel_long: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    ip_long = ip_df[["date", "IP_change"]].assign(
        branch="Total",
        indicator="IP_change",
        additional_info=None,
        value=lambda d: d["IP_change"],
        title_raw="IP_change",
        file="IndustrialProd.xlsx"
    )[["date","branch","indicator","additional_info","value","title_raw","file"]]

    out = pd.concat([panel_long, ip_long], ignore_index=True)
    out = out.sort_values(["date","branch","indicator"]).reset_index(drop=True)
    return out

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    # 1) Features (long)
    panel_long = load_all_features_long()
    panel_long.to_csv(OUT_CSV_PANEL_LONG, index=False, encoding="utf-8-sig")
    print(f"ifo_panel.csv mit {len(panel_long):,} Zeilen gespeichert")

    # 2) Target laden
    ip_path = TARGET_DIR / "IndustrialProd.xlsx"
    ip_df = load_ip_target(ip_path)
    ip_df.to_csv(OUT_TARGET_CSV, index=False, encoding="utf-8-sig")
    print(f"target.csv mit {len(ip_df):,} Monaten gespeichert")

    # 3) Panel inkl. IP_change
    panel_with_ip = add_ip_to_long(panel_long, ip_df)
    panel_with_ip.to_csv(OUT_CSV_PANEL_LONG_WITH_IP, index=False, encoding="utf-8-sig")
    print(f"panel_with_ip.csv mit {len(panel_with_ip):,} Zeilen gespeichert")
