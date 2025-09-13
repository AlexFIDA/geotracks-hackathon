# geosafety.py
# Usage:
#   python geosafety.py --in astana.csv --city astana --sample 200000 --use_grid 0
# Быстрый пайплайн: очистка, признаки, флаги аномалий, карта-слои, отчеты.

import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import folium

# ----------------------------- CLI ---------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True,
                    help="Входной CSV с колонками randomized_id,lat,lng,alt,spd,azm")
    ap.add_argument("--outdir", default="outputs", help="Куда сохранять результаты")
    ap.add_argument("--sample", type=int, default=0, help="Сэмпл строк для отладки (0 = весь датасет)")
    ap.add_argument("--use_grid", type=int, default=0, help="1 = плотность по сетке вместо kNN (быстрее)")
    ap.add_argument("--leaf_size", type=int, default=40, help="leaf_size для BallTree")
    ap.add_argument("--neighbors", type=int, default=10, help="k для kNN плотности")
    ap.add_argument("--city", default="astana", choices=["astana", "custom"], help="Бокс города")
    ap.add_argument("--bbox", default="50.9,51.3,71.2,71.6",
                    help="Если city=custom: lat_min,lat_max,lon_min,lon_max")
    return ap.parse_args()

# -------------------------- Utils ----------------------------------
EARTH_R = 6371000.0

def haversine_m_arr(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dphi = lat2 - lat1
    dlmb = lon2 - lon1
    a = np.sin(dphi/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlmb/2.0)**2
    return 2.0*EARTH_R*np.arcsin(np.sqrt(a))

def ang_diff_deg_arr(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def color_for(reason: str):
    if "gps_jump" in reason: return "black"
    if "speed_spike" in reason: return "purple"
    if "high_lateral_acc" in reason: return "red"
    if "sharp_turn" in reason: return "orange"
    if "sparse_area" in reason: return "blue"
    return "gray"

def speed_color(v_kmh: float):
    # простая градация для слоя "Скорость по точкам"
    if np.isnan(v_kmh): return "gray"
    if v_kmh < 20: return "green"
    if v_kmh < 40: return "blue"
    if v_kmh < 60: return "cadetblue"
    if v_kmh < 80: return "orange"
    if v_kmh < 120: return "red"
    return "darkred"

# -------------------------- Pipeline --------------------------------
def main():
    args = parse_args()
    t0 = time.time()
    ensure_outdir(args.outdir)

    # 1) Load
    df = pd.read_csv(args.infile)
    rename_map = {"randomized_id": "id", "lat": "lat", "lng": "lon", "alt": "alt", "spd": "spd", "azm": "azm"}
    df = df.rename(columns=rename_map)

    # Типы
    for c in ["lat", "lon", "alt", "spd", "azm", "id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Геобокс
    if args.city == "astana":
        lat_min, lat_max, lon_min, lon_max = 50.9, 51.3, 71.2, 71.6
    else:
        lat_min, lat_max, lon_min, lon_max = map(float, args.bbox.split(","))
    bbox = (df["lat"].between(lat_min, lat_max)) & (df["lon"].between(lon_min, lon_max))
    df = df[bbox].copy().reset_index(drop=True)

    # Проверка после bbox
    if df.empty:
        print("[error] после bbox нет точек. Проверьте --city/--bbox и входной CSV.")
        return

    # 3) Сэмпл
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # 4) Базовые фичи
    df["speed_ms"] = df["spd"].astype(np.float32)
    df["speed_kmh"] = df["speed_ms"] * 3.6

    # Сортировка по id (если временной колонки нет)
    df = df.sort_values(["id"]).reset_index(drop=True)

    # Сдвиги
    for col in ["lat", "lon", "azm", "speed_ms", "speed_kmh", "id"]:
        df[f"prev_{col}"] = df.groupby("id", sort=False)[col].shift(1)

    # Дистанция, углы
    df["step_m"] = haversine_m_arr(df["prev_lat"].values, df["prev_lon"].values,
                                   df["lat"].values, df["lon"].values).astype(np.float32)
    df.loc[df["id"] != df["prev_id"], "step_m"] = np.nan

    df["d_azm_deg"] = ang_diff_deg_arr(df["azm"].astype(np.float32),
                                       df["prev_azm"].astype(np.float32)).astype(np.float32)
    df.loc[df["id"] != df["prev_id"], "d_azm_deg"] = np.nan

    # Радиус поворота и поперечное ускорение
    theta_rad = np.deg2rad(df["d_azm_deg"].astype(np.float32))
    turn_radius = df["step_m"] / np.clip(theta_rad, 1e-6, None)
    df["turn_radius_m"] = turn_radius.astype(np.float32)
    df["a_lat"] = (df["speed_ms"] ** 2) / np.clip(df["turn_radius_m"], 1e-3, None)
    df.loc[(df["step_m"] < 5) | (df["speed_kmh"] < 5) | df["step_m"].isna(), "a_lat"] = 0.0
    df["a_lat"] = df["a_lat"].astype(np.float32)

    # 5) Локальные флаги
    cond_sharp_turn   = (df["d_azm_deg"] >= 45) & (df["speed_kmh"] >= 20) & (df["step_m"].between(5, 200))
    cond_high_lat_acc = df["a_lat"] >= 3.0
    cond_gps_jump     = df["step_m"] >= 1000
    cond_speed_spike  = df["speed_kmh"] >= 120

    # 6) Плотность: kNN или Grid
    cond_sparse = pd.Series(False, index=df.index)
    if args.use_grid:
        cell_lat = (df["lat"] * 500).round().astype(np.int32)
        cell_lon = (df["lon"] * 500).round().astype(np.int32)
        df["cell"] = (cell_lat.astype(str) + "_" + cell_lon.astype(str))
        counts = df["cell"].value_counts()
        df["cell_count"] = df["cell"].map(counts).astype(np.int32)
        thr = np.percentile(df["cell_count"], 2)
        cond_sparse = df["cell_count"] <= thr
    else:
        t_knn = time.time()
        coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=np.float32))
        nbrs = NearestNeighbors(
            n_neighbors=max(2, args.neighbors),
            algorithm="ball_tree",
            metric="haversine",
            leaf_size=args.leaf_size
        ).fit(coords)
        dists, _ = nbrs.kneighbors(coords, n_neighbors=max(2, args.neighbors))  # без n_jobs
        rK_m = dists[:, -1] * EARTH_R
        df["rK_m"] = rK_m.astype(np.float32)
        thr_sparse = np.nanpercentile(df["rK_m"], 98)
        cond_sparse = df["rK_m"] >= thr_sparse
        print(f"[perf] kNN density in {time.time()-t_knn:.2f}s")

    # 7) Итоговые флаги
    df["anomaly_reason"] = ""
    rules = [
        ("sharp_turn", cond_sharp_turn),
        ("high_lateral_acc", cond_high_lat_acc),
        ("gps_jump", cond_gps_jump),
        ("speed_spike", cond_speed_spike),
        ("sparse_area", cond_sparse),
    ]
    for name, cond in rules:
        idx = cond.fillna(False).values
        cur = df.loc[idx, "anomaly_reason"]
        df.loc[idx, "anomaly_reason"] = np.where(cur.eq(""), name, cur + "|" + name)
    df["is_anomaly"] = df["anomaly_reason"].ne("")

    # 8) Поездки
    trip_stats = (
        df.groupby("id")["is_anomaly"]
          .agg(["mean", "sum", "count"])
          .rename(columns={"mean": "anomaly_share", "sum": "anomaly_points", "count": "points"})
          .reset_index()
          .sort_values("anomaly_share", ascending=False)
    )
    trip_stats["is_bad_trip"] = trip_stats["anomaly_share"] >= 0.15

    # 9) Сохранение CSV
    point_cols = ["id", "lat", "lon", "speed_kmh", "d_azm_deg", "a_lat", "step_m", "anomaly_reason", "is_anomaly"]
    if "rK_m" in df.columns: point_cols.append("rK_m")
    if "cell_count" in df.columns: point_cols.append("cell_count")
    df.to_csv(os.path.join(args.outdir, "all_points_with_flags.csv"), index=False)
    df[df["is_anomaly"]][point_cols].to_csv(os.path.join(args.outdir, "point_anomalies.csv"), index=False)
    trip_stats.to_csv(os.path.join(args.outdir, "top_suspicious_trips.csv"), index=False)

    # 10) Единая карта со слоями
    center_lat = float(df["lat"].median()); center_lon = float(df["lon"].median())
    if not np.isfinite(center_lat) or not np.isfinite(center_lon):
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # Слой: Аномалии (точки)
    df_anom = df[df["is_anomaly"]].copy()
    fg_anom = folium.FeatureGroup(name="Резкие отклонения", show=True)
    if len(df_anom) > 0:
        sample_n = min(2000, len(df_anom))
        for _, r in df_anom.sample(sample_n, random_state=0).iterrows():
            folium.CircleMarker(
                location=[float(r.lat), float(r.lon)],
                radius=5,
                color=color_for(str(r.anomaly_reason)),
                fill=True, fill_opacity=0.8,
                popup=f"id={int(r.id) if pd.notna(r.id) else 'NA'} | {r.anomaly_reason}"
            ).add_to(fg_anom)
    fg_anom.add_to(m)

    # Слой: Маршруты подозрительных id
    bad_ids = set(trip_stats[trip_stats["is_bad_trip"]]["id"].head(5))
    fg_bad = folium.FeatureGroup(name="Необычные маршруты", show=True)
    for tid in bad_ids:
        seg = df[df["id"] == tid][["lat", "lon", "is_anomaly", "anomaly_reason"]].reset_index(drop=True)
        if len(seg) == 0:
            continue
        folium.PolyLine(seg[["lat", "lon"]].astype(float).values.tolist(), weight=3).add_to(fg_bad)
        for _, r in seg[seg["is_anomaly"]].iterrows():
            folium.CircleMarker(
                location=[float(r.lat), float(r.lon)],
                radius=5,
                color=color_for(str(r.anomaly_reason)),
                fill=True, fill_opacity=0.9
            ).add_to(fg_bad)
    fg_bad.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html = os.path.join(args.outdir, "combined_map.html")
    m.save(out_html)
    print(f"[out] {os.path.abspath(out_html)}")

    # 11) Короткий отчет
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"[done] points={total}, anomalies={anomalies} ({anomalies/max(total,1):.1%}), "
          f"bad_trips={(trip_stats['is_bad_trip']).sum()}")
    print(f"[out] saved to: {os.path.abspath(args.outdir)}; time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()