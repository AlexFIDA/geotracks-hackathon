import numpy as np
import pandas as pd
import folium
from sklearn.neighbors import NearestNeighbors
from branca.element import Element
from folium.plugins import AntPath

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

REASON_MAP = {
    "gps_jump":         ("Срыв GPS (скачок координаты)", "black"),
    "speed_spike":      ("Скачок скорости",               "purple"),
    "high_lateral_acc": ("Высокое боковое ускорение",     "red"),
    "sharp_turn":       ("Резкий поворот",                "orange"),
    "sparse_area":      ("Редкая зона (мало точек рядом)","blue"),
}

def _color_for(reason: str) -> str:
    code = str(reason).split("|")[0] if reason else ""
    return REASON_MAP.get(code, ("", "gray"))[1]

def pretty_reasons(reason: str) -> str:
    if not reason:
        return "нет"
    labels = []
    for code in str(reason).split("|"):
        labels.append(REASON_MAP.get(code, (code, "gray"))[0])
    return ", ".join(labels)

def speed_color(v_kmh: float):
    if np.isnan(v_kmh): return "gray"
    if v_kmh < 20: return "green"
    if v_kmh < 40: return "blue"
    if v_kmh < 60: return "cadetblue"
    if v_kmh < 80: return "orange"
    if v_kmh < 120: return "red"
    return "darkred"

def add_layers(m: folium.Map, df: pd.DataFrame, cfg: dict):
    x = df.rename(columns={"randomized_id": "id", "lng": "lon"}).copy()
    for c in ["lat", "lon", "alt", "spd", "azm", "id"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "id" not in x.columns:
        x["id"] = (x["lat"].diff().abs().gt(0.001) | x["lon"].diff().abs().gt(0.001)).cumsum()

    if cfg.get("city", "astana") == "astana":
        lat_min, lat_max, lon_min, lon_max = 50.9, 51.3, 71.2, 71.6
    else:
        lat_min, lat_max, lon_min, lon_max = map(float, cfg["bbox"].split(","))
    x = x[x["lat"].between(lat_min, lat_max) & x["lon"].between(lon_min, lon_max)].reset_index(drop=True)
    if x.empty:
        return

    sample_n = int(cfg.get("geo_sample", 0))
    if sample_n and sample_n < len(x):
        x = x.sample(sample_n, random_state=42).reset_index(drop=True)

    x["speed_ms"] = x["spd"].astype(np.float32)
    x["speed_kmh"] = x["speed_ms"] * 3.6
    x = x.sort_values(["id"]).reset_index(drop=True)
    for col in ["lat", "lon", "azm", "speed_ms", "speed_kmh", "id"]:
        x[f"prev_{col}"] = x.groupby("id", sort=False)[col].shift(1)

    x["step_m"] = _haversine_m(x["prev_lat"].values, x["prev_lon"].values,
                               x["lat"].values, x["lon"].values).astype(np.float32)
    x.loc[x["id"] != x["prev_id"], "step_m"] = np.nan
    x["d_azm_deg"] = _ang_diff_deg(x["azm"].astype(np.float32), x["prev_azm"].astype(np.float32)).astype(np.float32)
    x.loc[x["id"] != x["prev_id"], "d_azm_deg"] = np.nan

    theta = np.deg2rad(x["d_azm_deg"].astype(np.float32))
    x["turn_radius_m"] = (x["step_m"] / np.clip(theta, 1e-6, None)).astype(np.float32)
    x["a_lat"] = (x["speed_ms"]**2) / np.clip(x["turn_radius_m"], 1e-3, None)
    x.loc[(x["step_m"] < 5) | (x["speed_kmh"] < 5) | x["step_m"].isna(), "a_lat"] = 0.0
    x["a_lat"] = x["a_lat"].astype(np.float32)

    cond_sharp_turn   = (x["d_azm_deg"] >= 45) & (x["speed_kmh"] >= 20) & (x["step_m"].between(5, 200))
    cond_high_lat_acc = x["a_lat"] >= 3.0
    cond_gps_jump     = x["step_m"] >= 1000
    cond_speed_spike  = x["speed_kmh"] >= 120

    use_grid  = int(cfg.get("geo_use_grid", 0)) == 1
    neighbors = int(cfg.get("geo_neighbors", 10))
    leaf_size = int(cfg.get("geo_leaf_size", 40))
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
        coords = np.radians(x[["lat", "lon"]].to_numpy(dtype=np.float32))
        nbrs = NearestNeighbors(
            n_neighbors=max(2, neighbors), algorithm="ball_tree", metric="haversine", leaf_size=leaf_size
        ).fit(coords)
        dists, _ = nbrs.kneighbors(coords, n_neighbors=max(2, neighbors))
        x["rK_m"] = (dists[:, -1] * EARTH_R).astype(np.float32)
        thr_sparse = np.nanpercentile(x["rK_m"], 98)
        cond_sparse = x["rK_m"] >= thr_sparse

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

    trip_stats = (
        x.groupby("id")["is_anomaly"]
         .agg(["mean", "sum", "count"])
         .rename(columns={"mean": "anomaly_share", "sum": "anomaly_points", "count": "points"})
         .reset_index()
         .sort_values("anomaly_share", ascending=False)
    )
    trip_stats["is_bad_trip"] = trip_stats["anomaly_share"] >= 0.15

    # 1) Точки-аномалии
    fg_anom = folium.FeatureGroup(name="Geosafety - Резкие отклонения", show=False)
    df_anom = x[x["is_anomaly"]]
    if not df_anom.empty:
        n = min(int(cfg.get("geo_anom_points", 2000)), len(df_anom))
        for _, r in df_anom.sample(n, random_state=0).iterrows():
            folium.CircleMarker(
                [float(r["lat"]), float(r["lon"])], radius=5,
                color=_color_for(str(r["anomaly_reason"])),
                fill=True, fill_opacity=0.8,
                popup=f"Причина: {pretty_reasons(r['anomaly_reason'])}"
            ).add_to(fg_anom)
    fg_anom.add_to(m)

    # 2) Необычные маршруты: прямая A→B + маркеры A/B
    fg_ab = folium.FeatureGroup(name="Geosafety - Необычные маршруты (A→B)", show=False)
    grp = x.groupby("id", sort=False)
    se = grp[["lat", "lon"]].first().rename(columns={"lat": "start_lat", "lon": "start_lon"}).join(
         grp[["lat", "lon"]].last().rename(columns={"lat": "end_lat", "lon": "end_lon"})
    ).reset_index()

    for tid in set(trip_stats[trip_stats["is_bad_trip"]]["id"].head(int(cfg.get("geo_bad_trips", 5)))):
        row = se[se["id"] == tid]
        if row.empty:
            continue
        s_lat, s_lon = float(row["start_lat"].iloc[0]), float(row["start_lon"].iloc[0])
        e_lat, e_lon = float(row["end_lat"].iloc[0]),   float(row["end_lon"].iloc[0])

        folium.PolyLine([[s_lat, s_lon], [e_lat, e_lon]],
                        weight=4, opacity=0.8, tooltip="Необычный маршрут A→B").add_to(fg_ab)
        AntPath([[s_lat, s_lon], [e_lat, e_lon]], delay=800, weight=4, dash_array=[10, 20]).add_to(fg_ab)
        folium.CircleMarker([s_lat, s_lon], radius=6, color="black", fill=True, fill_opacity=1.0,
                            popup="Старт (A)").add_to(fg_ab)
        folium.CircleMarker([e_lat, e_lon], radius=6, color="darkblue", fill=True, fill_opacity=1.0,
                            popup="Финиш (B)").add_to(fg_ab)
    fg_ab.add_to(m)

    # 3) Легенда, привязка к fg_anом
    legend_html = """
    <div id="legend-anoms" style="
      position: fixed; bottom: 20px; left: 20px; z-index: 9999;
      background: white; padding: 10px 12px; border: 1px solid #999; font-size: 12px;">
      <b>Geosafety - Резкие отклонения</b><br>
      <span style="color:black;">■</span> Срыв GPS<br>
      <span style="color:purple;">■</span> Скачок скорости<br>
      <span style="color:red;">■</span> Высокое боковое ускорение<br>
      <span style="color:orange;">■</span> Резкий поворот<br>
      <span style="color:blue;">■</span> Редкая зона
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    toggle_js = """
    <script>
      var mapRef = MAPNAME;
      var fg = FGLAYER;
      function setLegend(show){
        var el = document.getElementById('legend-anoms');
        if (!el) return;
        el.style.display = show ? 'block' : 'none';
      }
      setLegend(mapRef.hasLayer(fg));
      mapRef.on('overlayadd',   function(e){ if (e.layer === fg) setLegend(true);  });
      mapRef.on('overlayremove',function(e){ if (e.layer === fg) setLegend(false); });
    </script>
    """.replace("MAPNAME", m.get_name()).replace("FGLAYER", fg_anom.get_name())
    m.get_root().html.add_child(Element(toggle_js))