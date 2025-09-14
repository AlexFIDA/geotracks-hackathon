# build_map.py
from pathlib import Path
import time
import folium

from common import load_df, make_map
from layers import make_heatmap, iros, geosafety

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

CFG = {
    "city": "astana",
    "bbox": "50.9,51.3,71.2,71.6",
    "heat_sample_limit": 120_000, "heat_sample_take": 100_000,
    "speed_points": 2000, "top_routes": 100,
    # GeoSafety — безопасные дефолты
    "geo_sample": 150_000,     # 0 = весь датасет
    "geo_use_grid": 1,         # 1 = быстро, без kNN
    "geo_neighbors": 10, "geo_leaf_size": 40,
    "geo_anom_points": 2000, "geo_bad_trips": 5,
}

def _add(mod, m, df, cfg):
    t = time.time()
    name = getattr(mod, "__name__", str(mod))
    print(f"[add] {name} ...", flush=True)
    mod.add_layers(m, df, cfg)
    print(f"[ok ] {name} in {time.time()-t:.1f}s", flush=True)

def main():
    print("[load] data...", flush=True)
    df = load_df()
    print(f"[ok  ] rows={len(df)}", flush=True)

    m = make_map(df)

    # Подмешиваем слои
    for mod in (make_heatmap, iros, geosafety):
        _add(mod, m, df, CFG)

    from folium import LayerControl
    LayerControl(collapsed=False).add_to(m)

    out_file = OUT / "heatmap.html"
    m.save(str(out_file))
    print(f"[done] {out_file.resolve()}", flush=True)

if __name__ == "__main__":
    main()