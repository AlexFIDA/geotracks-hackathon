from folium.plugins import HeatMap
from pathlib import Path
import folium
import numpy as np
import pandas as pd

def add_layers(m, df, cfg):
    # ---- Heatmap (выкл по умолчанию)
    fg = folium.FeatureGroup(name="Heatmap — все точки", show=False)
    sample = df[["lat", "lng"]]
    if len(sample) > cfg.get("heat_sample_limit", 120_000):
        sample = sample.sample(n=cfg.get("heat_sample_take", 100_000), random_state=42)
    HeatMap(sample.values.tolist(), radius=10, blur=15, min_opacity=0.2).add_to(fg)
    fg.add_to(m)

    # ---- Выводы/каталоги
    outdir = Path(cfg.get("outdir", "outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Matplotlib без GUI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1) Гистограмма скоростей
    s = pd.to_numeric(df["spd"], errors="coerce")
    speed_kmh = (s * 3.6).clip(lower=0, upper=120).dropna()
    plt.figure()
    speed_kmh.hist(bins=40)
    plt.title("Скорость, км/ч"); plt.xlabel("км/ч"); plt.ylabel("кол-во точек")
    plt.tight_layout(); plt.savefig(outdir / "speed_hist.png"); plt.close()

    # ---- Подготовка к углам/ускорению
    x = df[["lat", "lng", "azm", "spd"]].copy()
    x["azm"] = np.mod(pd.to_numeric(x["azm"], errors="coerce"), 360.0)
    x["spd_ms"] = pd.to_numeric(x["spd"], errors="coerce")

    # Fallback для идентификатора трека
    if "randomized_id" in df.columns:
        gid = df["randomized_id"]
    else:
        # простой эвристический id: новый id при заметном скачке координат
        gid = (x["lat"].diff().abs().gt(0.001) | x["lng"].diff().abs().gt(0.001)).cumsum()
    x["gid"] = gid

    # Сдвиги по трекам
    x["prev_lat"] = x.groupby("gid", sort=False)["lat"].shift(1)
    x["prev_lng"] = x.groupby("gid", sort=False)["lng"].shift(1)
    x["prev_azm"] = x.groupby("gid", sort=False)["azm"].shift(1)

    # ---- 2) |Δазимут|
    def ang_diff_deg(a, b):
        return np.abs((a - b + 180.0) % 360.0 - 180.0)

    x["d_azm_deg"] = ang_diff_deg(x["azm"], x["prev_azm"])
    d_azm = x["d_azm_deg"].astype(float).clip(0, 180).dropna()

    plt.figure()
    d_azm.hist(bins=36)  # по 5°
    plt.title("|Δазимут|, градусы"); plt.xlabel("градусы"); plt.ylabel("кол-во точек")
    plt.tight_layout(); plt.savefig(outdir / "turn_hist.png"); plt.close()

    # ---- 3) Поперечное ускорение a_lat
    EARTH_R = 6_371_000.0
    def hav_m(lat1, lon1, lat2, lon2):
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dphi = lat2 - lat1
        dlmb = lon2 - lon1
        a = np.sin(dphi/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlmb/2.0)**2
        return 2.0*EARTH_R*np.arcsin(np.sqrt(a))

    step_m = hav_m(x["prev_lat"], x["prev_lng"], x["lat"], x["lng"])
    theta = np.deg2rad(x["d_azm_deg"])
    R = step_m / np.clip(theta, 1e-6, None)
    a_lat = (x["spd_ms"]**2) / np.clip(R, 1e-3, None)
    a_lat = pd.Series(a_lat, dtype="float64").replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=5).dropna()

    plt.figure()
    a_lat.hist(bins=40)
    plt.title("Поперечное ускорение a_lat, м/с²"); plt.xlabel("м/с²"); plt.ylabel("кол-во точек")
    plt.tight_layout(); plt.savefig(outdir / "a_lat_hist.png"); plt.close()
