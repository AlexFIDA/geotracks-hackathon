import pandas as pd, numpy as np
import folium
from folium.plugins import HeatMap
from pathlib import Path
import matplotlib.pyplot as plt

csv_path = Path("Data/geo_locations_astana_hackathon")
out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path).drop_duplicates()
df.loc[df["spd"] < 0, "spd"] = np.nan
df["speed_kmh"] = df["spd"] * 3.6
df["azm"] = np.mod(df["azm"].astype(float), 360.0)

# график
df["speed_kmh"].dropna().clip(upper=120).hist(bins=40)
plt.title("Скорость, км/ч"); plt.xlabel("км/ч"); plt.ylabel("кол-во точек"); plt.tight_layout()
plt.savefig(out/"speed_hist.png"); plt.close()

# карта
center = [df["lat"].median(), df["lng"].median()]
m = folium.Map(location=center, zoom_start=12, control_scale=True)
sample = df[["lat","lng"]].sample(n=100_000, random_state=42) if len(df) > 120_000 else df[["lat","lng"]]
HeatMap(sample.values.tolist(), radius=10, blur=15, min_opacity=0.2).add_to(m)
m.save(str(out/"heatmap.html"))

print("OK ->", out/"heatmap.html")
