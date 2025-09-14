import numpy as np
import pandas as pd
import folium
from sklearn.neighbors import NearestNeighbors

EARTH_R = 6371000.0

def _haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dphi = lat2 - lat1
    dlmb = lon2 - lon1
    a = np.sin(dphi/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlmb/2.0)**2
    return 2.0*EARTH_R*np.arcsin(np.sqrt(a))

def _ang_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

def _color_for(reason: str):
    if "gps_jump" in reason: return "black"
    if "speed_spike" in reason: return "purple"
    if "high_lateral_acc" in reason: return "red"
    if "sharp_turn" in reason: return "orange"
    if "sparse_area" in reason: return "blue"
    return "gray"

def add_layers(m: folium.Map, df: pd.DataFrame, cfg: dict):
    # --- подготовка данных
    x = df.rename(columns={"randomized_id":"id","lng":"lon"}).copy()
    for c in ["lat","lon","alt","spd","azm","id"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    # bbox по Астане или кастом
    if cfg.get("city","astana") == "astana":
        lat_min, lat_max, lon_min, lon_max = 50.9, 51.3, 71.2, 71.6
    else:
        lat_min, lat_max, lon_min, lon_max = map(float, cfg["bbox"].split(","))
    box = x["lat"].between(lat_min, lat_max) & x["lon"].between(lon_min, lon_max)
    x = x.loc[box].reset_index(drop=True)
    if x.empty:
        return  # нечего добавлять

    # сэмпл для ускорения
    sample_n = int(cfg.get("geo_sample", 0))
    if sample_n and sample_n < len(x):
        x = x.sample(sample_n, random_state=42).reset_index(drop=True)

    # фичи
    x["speed_ms"]  = x["spd"].astype(np.float32)
    x["speed_kmh"] = x["speed_ms"] * 3.6
    x = x.sort_values(["id"]).reset_index(drop=True)
    for col in ["lat","lon","azm","speed_ms","speed_kmh","id"]:
        x[f"prev_{col}"] = x.groupby("id", sort=False)[col].shift(1)

    x["step_m"]   = _haversine_m(x["prev_lat"].values, x["prev_lon"].values,
                                 x["lat"].values,      x["lon"].values).astype(np.float32)
    x.loc[x["id"] != x["prev_id"], "step_m"] = np.nan

    x["d_azm_deg"] = _ang_diff_deg(x["azm"].astype(np.float32),
                                   x["prev_azm"].astype(np.float32)).astype(np.float32)
    x.loc[x["id"] != x["prev_id"], "d_azm_deg"] = np.nan

    theta = np.deg2rad(x["d_azm_deg"].astype(np.float32))
    turn_radius = x["step_m"] / np.clip(theta, 1e-6, None)
    x["turn_radius_m"] = turn_radius.astype(np.float32)
    x["a_lat"] = (x["speed_ms"]**2) / np.clip(x["turn_radius_m"], 1e-3, None)
    x.loc[(x["step_m"] < 5) | (x["speed_kmh"] < 5) | x["step_m"].isna(), "a_lat"] = 0.0
    x["a_lat"] = x["a_lat"].astype(np.float32)

    # локальные флаги
    cond_sharp_turn   = (x["d_azm_deg"] >= 45) & (x["speed_kmh"] >= 20) & (x["step_m"].between(5,200))
    cond_high_lat_acc = x["a_lat"] >= 3.0
    cond_gps_jump     = x["step_m"] >= 1000
    cond_speed_spike  = x["speed_kmh"] >= 120

    # плотность: kNN или grid
    use_grid   = int(cfg.get("geo_use_grid", 0)) == 1
    neighbors  = int(cfg.get("geo_neighbors", 10))
    leaf_size  = int(cfg.get("geo_leaf_size", 40))
    cond_sparse = pd.Series(False, index=x.index)

    if use_grid:
        cell_lat = (x["lat"] * 500).round().astype(np.int32)
        cell_lon = (x["lon"] * 500).round().astype(np.int32)
        x["cell"] = (cell_lat.astype(str) + "_" + cell_lon.astype(str))
        counts = x["cell"].value_counts()
        x["cell_count"] = x["cell"].map(counts).astype(np.int32)
        thr = np.percentile(x["cell_count"], 2)
        cond_sparse = x["cell_count"] <= thr
    else:
        coords = np.radians(x[["lat","lon"]].to_numpy(dtype=np.float32))
        nbrs = NearestNeighbors(
            n_neighbors=max(2, neighbors),
            algorithm="ball_tree",
            metric="haversine",
            leaf_size=leaf_size
        ).fit(coords)
        dists, _ = nbrs.kneighbors(coords, n_neighbors=max(2, neighbors))
        rK_m = dists[:, -1] * EARTH_R
        x["rK_m"] = rK_m.astype(np.float32)
        thr_sparse = np.nanpercentile(x["rK_m"], 98)
        cond_sparse = x["rK_m"] >= thr_sparse

    # итоговые флаги
    x["anomaly_reason"] = ""
    for name, cond in [
        ("sharp_turn", cond_sharp_turn),
        ("high_lateral_acc", cond_high_lat_acc),
        ("gps_jump", cond_gps_jump),
        ("speed_spike", cond_speed_spike),
        ("sparse_area", cond_sparse),
    ]:
        idx = cond.fillna(False).values
        cur = x.loc[idx, "anomaly_reason"]
        x.loc[idx, "anomaly_reason"] = np.where(cur.eq(""), name, cur + "|" + name)
    x["is_anomaly"] = x["anomaly_reason"].ne("")

    # сводка по поездкам
    trip_stats = (
        x.groupby("id")["is_anomaly"]
         .agg(["mean","sum","count"])
         .rename(columns={"mean":"anomaly_share","sum":"anomaly_points","count":"points"})
         .reset_index()
    )
    trip_stats["is_bad_trip"] = trip_stats["anomaly_share"] >= 0.15

    # --- СЛОИ НА КАРТУ ---

    # 1) Точки-анomalии
    fg_anom = folium.FeatureGroup(name="GeoSafety: резкие отклонения", show=True)
    df_anom = x[x["is_anomaly"]]
    if not df_anom.empty:
        n = min(int(cfg.get("geo_anom_points", 2000)), len(df_anom))
        for _, r in df_anom.sample(n, random_state=0).iterrows():
            folium.CircleMarker(
                [float(r.lat), float(r.lon)], radius=5,
                color=_color_for(str(r.anomaly_reason)),
                fill=True, fill_opacity=0.8,
                popup=f"id={int(r.id) if pd.notna(r.id) else 'NA'} | {r.anomaly_reason}"
            ).add_to(fg_anom)
    fg_anom.add_to(m)

    # 2) Маршруты самых проблемных id
    fg_bad = folium.FeatureGroup(name="GeoSafety: необычные маршруты", show=True)
    bad_ids = set(trip_stats[trip_stats["is_bad_trip"]]["id"].head(int(cfg.get("geo_bad_trips", 5))))
    for tid in bad_ids:
        seg = x[x["id"] == tid][["lat","lon","is_anomaly","anomaly_reason"]]
        if seg.empty:
            continue
        folium.PolyLine(seg[["lat","lon"]].astype(float).values.tolist(), weight=3).add_to(fg_bad)
        for _, r in seg[seg["is_anomaly"]].iterrows():
            folium.CircleMarker([float(r.lat), float(r.lon)],
                                radius=5, color=_color_for(str(r.anomaly_reason)),
                                fill=True, fill_opacity=0.9).add_to(fg_bad)
    fg_bad.add_to(m)
