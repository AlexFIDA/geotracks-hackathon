from folium.plugins import HeatMap

def add_layers(m, df, cfg):
    sample = df[["lat","lng"]]
    if len(sample) > cfg.get("heat_sample_limit", 120_000):
        sample = sample.sample(n=cfg.get("heat_sample_take", 100_000), random_state=42)
    HeatMap(sample.values.tolist(),
            radius=10, blur=15, min_opacity=0.2,
            name="Heatmap — все точки").add_to(m)
