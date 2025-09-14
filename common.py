from pathlib import Path
import pandas as pd, numpy as np
import folium

ROOT = Path(__file__).resolve().parent
DATA_BASE = ROOT / "Data" / "geo_locations_astana_hackathon"

def _resolve_path(base: Path) -> Path:
    cands = [base, Path(str(base)+".csv"), Path(str(base)+".parquet"), Path(str(base)+".gz")]
    for p in cands:
        if p.exists(): return p
    raise FileNotFoundError(f"Не найден файл данных. Проверены: {[str(p) for p in cands]}")

def load_df(path_base: Path = DATA_BASE) -> pd.DataFrame:
    p = _resolve_path(path_base)
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    df = df.drop_duplicates()
    df.loc[df["spd"] < 0, "spd"] = np.nan
    df["spd_kmh"] = df["spd"] * 3.6
    df["azm"] = np.mod(df["azm"].astype(float), 360.0)
    return df

def make_map(df: pd.DataFrame) -> folium.Map:
    center = [df["lat"].median(), df["lng"].median()]
    return folium.Map(location=center, zoom_start=12, control_scale=True)
