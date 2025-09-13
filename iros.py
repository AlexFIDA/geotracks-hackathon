import pandas as pd
import folium
from folium.plugins import HeatMap
import random

file = r"C:\Users\Рамазан\Desktop\geo_locations_astana_hackathon"

if __name__ == '__main__':

    # Загружаем и чистим
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df[(df["spd"] >= 0) & (df["spd"] < 150)]
    df["spd_kmh"] = df["spd"] * 3.6

    # Получаем старты и финиши
    starts = df.groupby("randomized_id").first().reset_index()
    ends = df.groupby("randomized_id").last().reset_index()

    # --- Карта
    m = folium.Map(location=[51.1694, 71.4491], zoom_start=12)

    # --- Слой 1: тепловая карта всех поездок
    heat_all = HeatMap(df[["lat", "lng"]].values.tolist(),
                       name="Heatmap — все поездки", radius=6, min_opacity=0.4)

    # --- Слой 2: тепловая карта посадок
    heat_start = HeatMap(starts[["lat", "lng"]].values.tolist(),
                         name="Heatmap — посадки", radius=6, min_opacity=0.4)

    # --- Слой 3: тепловая карта высадок
    heat_end = HeatMap(ends[["lat", "lng"]].values.tolist(),
                       name="Heatmap — высадки", radius=6, min_opacity=0.4)

    # --- Слой 4: точки по скорости (красный/оранжевый/зелёный)
    speed_layer = folium.FeatureGroup(name="Точки по скорости")

    def speed_color(speed):
        if speed < 20:   # пробка
            return "red"
        elif speed < 60: # городской режим
            return "orange"
        else:            # трасса
            return "green"

    for _, row in df.sample(2000).iterrows():  # ограничим до 2000 точек
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=3,
            color=speed_color(row["spd_kmh"]),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['spd_kmh']:.1f} км/ч"
        ).add_to(speed_layer)



    # --- Слой 6: ТОП маршруты (толщина = частота)
    routes = pd.merge(starts, ends, on="randomized_id", suffixes=("_start", "_end"))
    routes["start_lat_round"] = routes["lat_start"].round(3)
    routes["start_lng_round"] = routes["lng_start"].round(3)
    routes["end_lat_round"] = routes["lat_end"].round(3)
    routes["end_lng_round"] = routes["lng_end"].round(3)

    top_routes = routes.groupby(
        ["start_lat_round","start_lng_round","end_lat_round","end_lng_round"]
    ).size().reset_index(name="count").sort_values("count", ascending=False).head(100)

    top_routes_layer = folium.FeatureGroup(name="ТОП маршруты")
    for _, row in top_routes.iterrows():
        coords = [
            [row["start_lat_round"], row["start_lng_round"]],
            [row["end_lat_round"], row["end_lng_round"]]
        ]
        folium.PolyLine(coords,
                        color="yellow",
                        weight=2 + row["count"] / top_routes["count"].max() * 8,
                        opacity=0.7,
                        tooltip=f"Маршрут: {row['count']} поездок").add_to(top_routes_layer)

    # --- Слой 7: узкие места (пробки)
    df["lat_round"] = df["lat"].round(4)
    df["lng_round"] = df["lng"].round(4)
    speed_by_point = df.groupby(["lat_round","lng_round"])["spd_kmh"].mean().reset_index()
    bottlenecks = speed_by_point[speed_by_point["spd_kmh"] < 10]

    bottleneck_layer = folium.FeatureGroup(name="Узкие места (пробки)")
    for _, row in bottlenecks.iterrows():
        folium.CircleMarker(
            location=[row["lat_round"], row["lng_round"]],
            radius=6,
            color="red",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"Средняя скорость: {row['spd_kmh']:.1f} км/ч"
        ).add_to(bottleneck_layer)

    # --- Добавляем все слои на карту
    m.add_child(heat_all)
    m.add_child(heat_start)
    m.add_child(heat_end)
    m.add_child(speed_layer)
    m.add_child(top_routes_layer)
    m.add_child(bottleneck_layer)

    folium.LayerControl().add_to(m)

    # --- Сохраняем
    m.save("itog.html")
    print("✅ Карта сохранена в itog.html")
