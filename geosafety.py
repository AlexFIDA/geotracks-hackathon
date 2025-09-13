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
from folium.plugins import AntPath

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

# Человеко-понятные названия причин + цвета
REASON_MAP = {
    "gps_jump":         ("Срыв GPS (скачок координаты)", "black"),
    "speed_spike":      ("Скачок скорости",               "purple"),
    "high_lateral_acc": ("Высокое боковое ускорение",     "red"),
    "sharp_turn":       ("Резкий поворот",                "orange"),
    "sparse_area":      ("Редкая зона (мало точек рядом)","blue"),
}

def color_for(reason: str) -> str:
    code = str(reason).split("|")[0] if reason else ""
    return REASON_MAP.get(code, ("", "gray"))[1]

def pretty_reasons(reason: str) -> str:
    if not reason:
        return "нет"
    labels = []
    for code in str(reason).split("|"):
        labels.append(REASON_MAP.get(code, (code, "gray"))[0])
    return ", ".join(labels)

def track_label(val) -> str:
    try:
        return f"Трек №{int(val)}"
    except Exception:
        return "Трек (без номера)"

def speed_color(v_kmh: float):
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

    if df.empty:
        print("[error] после bbox нет точек. Проверьте --city/--bbox и входной CSV.")
        return

    # 3) Сэмпл
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # 4) Базовые фичи
    df["speed_ms"] = df["spd"].astype(np.float32)
    df["speed_kmh"] = df["speed_ms"] * 3.6

    # Сортировка по id (без времени)
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
        dists, _ = nbrs.kneighbors(coords, n_neighbors=max(2, args.neighbors))
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
    ensure_outdir(args.outdir)
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
                location=[float(r["lat"]), float(r["lon"])],
                radius=5,
                color=color_for(str(r["anomaly_reason"])),
                fill=True, fill_opacity=0.8,
                popup=f"{track_label(r['id'])} | Причина: {pretty_reasons(r['anomaly_reason'])}"
            ).add_to(fg_anom)
    fg_anom.add_to(m)

    # Слой: Маршруты подозрительных треков — ЛИНИЯ от старта до финиша
    bad_ids = set(trip_stats[trip_stats["is_bad_trip"]]["id"].head(5))

    # старт/финиш по каждому треку (без явного времени берем first/last)
    grp = df.groupby("id", sort=False)
    starts = grp[["lat", "lon"]].first().rename(columns={"lat": "start_lat", "lon": "start_lon"})
    ends   = grp[["lat", "lon"]].last().rename(columns={"lat": "end_lat",  "lon": "end_lon"})
    se = starts.join(ends).reset_index()

    fg_bad = folium.FeatureGroup(name="Необычные маршруты (A→B)", show=True)
    for tid in bad_ids:
        row = se[se["id"] == tid]
        if row.empty:
            continue
        s_lat, s_lon = float(row["start_lat"].iloc[0]), float(row["start_lon"].iloc[0])
        e_lat, e_lon = float(row["end_lat"].iloc[0]),   float(row["end_lon"].iloc[0])

        # Прямая от A к B
        folium.PolyLine([[s_lat, s_lon], [e_lat, e_lon]],
                        weight=4, opacity=0.8, tooltip=track_label(tid)).add_to(fg_bad)

        # Точка A (старт)
        folium.CircleMarker([s_lat, s_lon], radius=6, color="black", fill=True, fill_opacity=1.0,
                            popup=f"{track_label(tid)} — Старт (A)").add_to(fg_bad)
        # Точка B (финиш)
        folium.CircleMarker([e_lat, e_lon], radius=6, color="darkblue", fill=True, fill_opacity=1.0,
                            popup=f"{track_label(tid)} — Финиш (B)").add_to(fg_bad)
    AntPath([[s_lat, s_lon], [e_lat, e_lon]], delay=800, weight=4, dash_array=[10, 20]).add_to(fg_bad)
    fg_bad.add_to(m)
    

    # Легенда
    legend = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
    background: white; padding: 10px 12px; border: 1px solid #999; font-size: 12px;">
    <b>Аномалии</b><br>
    <span style="color:black;">■</span> Срыв GPS<br>
    <span style="color:purple;">■</span> Скачок скорости<br>
    <span style="color:red;">■</span> Высокое боковое ускорение<br>
    <span style="color:orange;">■</span> Резкий поворот<br>
    <span style="color:blue;">■</span> Редкая зона
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    out_html = os.path.join(args.outdir, "combined_map.html")
    m.save(out_html)
    print(f"[out] {os.path.abspath(out_html)}")

    # 11) Короткий отчет
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    share = anomalies / total if total else 0.0
    print(f"[done] points={total}, anomalies={anomalies} ({share:.1%}), "
          f"bad_trips={(trip_stats['is_bad_trip']).sum()}")
    print(f"[out] saved to: {os.path.abspath(args.outdir)}; time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()