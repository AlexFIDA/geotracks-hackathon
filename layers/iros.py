import folium
import pandas as pd
from folium.plugins import HeatMap

def add_layers(m, df, cfg):
    # старт/финиш
    starts = df.groupby("randomized_id").first().reset_index()
    ends   = df.groupby("randomized_id").last().reset_index()

    # 1–3: теплокарты (выключены по умолчанию)
    for name, data, r in [
        ("Heatmap — все поездки", df, 6),
        ("Heatmap — посадки",     starts, 6),
        ("Heatmap — высадки",     ends, 6),
    ]:
        fg = folium.FeatureGroup(name=name, show=False)
        HeatMap(data[["lat","lng"]].values.tolist(), radius=r, min_opacity=0.4).add_to(fg)
        fg.add_to(m)

    # 4: точки по скорости
    def speed_color(v): return "red" if v < 20 else ("orange" if v < 60 else "green")
    speed_layer = folium.FeatureGroup(name="Точки по скорости", show=False)
    sample_n = cfg.get("speed_points", 2000)
    src = df.dropna(subset=["spd_kmh"])
    for _, row in src.sample(min(sample_n, len(src)), random_state=42).iterrows():
        folium.CircleMarker(
            [row["lat"], row["lng"]], radius=3,
            color= speed_color(row["spd_kmh"]),
            fill=True, fill_opacity=0.7,
            popup=f"{row['spd_kmh']:.1f} км/ч"
        ).add_to(speed_layer)
    speed_layer.add_to(m)

    # 6: ТОП маршруты
    routes = pd.merge(starts, ends, on="randomized_id", suffixes=("_start","_end"))
    for c in ["lat_start","lng_start","lat_end","lng_end"]:
        routes[c + "_r"] = routes[c].round(3)
    grp = routes.groupby(["lat_start_r","lng_start_r","lat_end_r","lng_end_r"]).size()
    top_routes = grp.reset_index(name="count").sort_values("count", ascending=False)\
                    .head(cfg.get("top_routes", 100))
    top_routes_layer = folium.FeatureGroup(name="ТОП маршруты", show=False)
    mx = top_routes["count"].max() if len(top_routes) else 1
    for _, r in top_routes.iterrows():
        folium.PolyLine([[r["lat_start_r"], r["lng_start_r"]],
                         [r["lat_end_r"],   r["lng_end_r"]]],
                        color="yellow", weight=2 + r["count"]/mx*8,
                        opacity=0.7, tooltip=f"Маршрут: {r['count']} поездок").add_to(top_routes_layer)
    top_routes_layer.add_to(m)

    # 7: узкие места
    g = df.assign(lat_round=df["lat"].round(4), lng_round=df["lng"].round(4)) \
          .groupby(["lat_round","lng_round"])["spd_kmh"].mean().reset_index()
    b_layer = folium.FeatureGroup(name="Узкие места", show=False)
    for _, r in g[g["spd_kmh"] < 10].iterrows():
        folium.CircleMarker([r["lat_round"], r["lng_round"]],
                            radius=6, color="red", fill=True, fill_opacity=0.6,
                            tooltip=f"Средняя скорость: {r['spd_kmh']:.1f} км/ч").add_to(b_layer)
    b_layer.add_to(m)
