# load_data.py
import pandas as pd, re, glob, pathlib, numpy as np

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "RawData"
OUT_CSV  = BASE_DIR / "ifo_panel.csv"

PAT = re.compile(r'^(?P<indicator>.+?)\s*\((?P<info>[A-Z])\)\s*(?P<branch>.+?)(?:\s+BD\s+SBR)?$')

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

def parse_month(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)) and not np.isnan(x):
        return pd.to_datetime(x, unit='d', origin='1899-12-30', errors='coerce')
    s = str(x).strip().replace('.', '/').replace('-', '/')
    if re.fullmatch(r'\d{2}/\d{4}', s):
        return pd.to_datetime(s, format='%m/%Y', errors='coerce')
    if re.fullmatch(r'\d{4}/\d{2}', s):
        y, m = s.split('/')
        return pd.to_datetime(f'{y}-{m}-01', errors='coerce')
    dt = pd.to_datetime(s, errors='coerce')
    if pd.notna(dt):
        return pd.Timestamp(dt.year, dt.month, 1)
    return pd.NaT

def load_file(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=2, engine="openpyxl")
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = df["date"].map(parse_month)

    value_cols = [c for c in df.columns[1:] if not df[c].isna().all()]
    df = df[["date"] + value_cols]

    long = df.melt(id_vars="date", var_name="title_raw", value_name="value")
    # hier: **erstmal kein dropna auf value** → Missings bleiben
    parts = long["title_raw"].map(parse_title).apply(pd.Series)
    out = pd.concat([long, parts], axis=1)

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["file"]  = path.name

    out["branch"]    = out["branch"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["indicator"] = out["indicator"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return out[["date","branch","indicator","additional_info","value","title_raw","file"]]

def load_all() -> pd.DataFrame:
    files = sorted(map(pathlib.Path, glob.glob(str(DATA_DIR / "bdi1a_*.xlsx"))))
    if not files:
        raise FileNotFoundError(f"keine .xlsx unter {DATA_DIR}/bdi1a_*.xlsx")
    panel = pd.concat([load_file(f) for f in files], ignore_index=True)
    panel = panel.sort_values(["date","branch","indicator"]).reset_index(drop=True)
    return panel

if __name__ == "__main__":
    df = load_all()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"ok → {OUT_CSV} | rows: {len(df)} | series:",
          df[["branch","indicator","additional_info"]].drop_duplicates().shape[0])
