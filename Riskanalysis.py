import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "."
MAP_IMAGE = "your_map.png"
OUTPUT_DIR = "map_heatmaps_by_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE = 300
HEAT_CLIP_QUANTILE = 0.98
HEAT_GAMMA = 0.45
# Use full data extent for better map alignment by default.
EXTENT_LOW_Q = 0.005
EXTENT_HIGH_Q = 0.995
SMOOTH_PASSES = 3
MIN_VISIBLE_FRACTION = 0.01
CIRCLE_RADIUS = 6
CIRCLE_SIGMA = 2.2
# Calibrated bounds from Untitled-1.py for this basemap.
# Set to None to fall back to auto bounds from data quantiles.
MAP_BOUNDS = {
    "min_lat": 42.977589,   # LAT_BOT
    "max_lat": 43.087394,   # LAT_TOP
    "min_lon": -76.217281,  # LON_LEFT
    "max_lon": -76.079200,  # LON_RIGHT
}
SPECIAL_SCALE = {
    # Fewer points: boost low-intensity visibility.
    "Vacant_Properties.csv": {
        "clip_q": 0.985,
        "gamma": 1.00,
        "alpha": 0.52,
        "bins": 320,
        "show_points": False,
        "radius": 8,
        "sigma": 3.0,
    },
    "Unfit_Properties.csv": {
        "clip_q": 0.99,
        "gamma": 1.00,
        "alpha": 0.52,
        "bins": 340,
        "show_points": False,
        "radius": 9,
        "sigma": 3.2,
    },
    "Crime_Data_2024_Combined.csv": {
        "clip_q": 0.99,
        "gamma": 1.00,
        "alpha": 0.50,
        "bins": 180,
        "show_points": False,
        "radius": 4,
        "sigma": 1.8,
    },
    "Crime_Data_2023_Combined.csv": {
        "clip_q": 0.99,
        "gamma": 1.00,
        "alpha": 0.50,
        "bins": 180,
        "show_points": False,
        "radius": 4,
        "sigma": 1.8,
    },
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def pick_lat_lon_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "long", "lon", "x"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col


def extract_points(df: pd.DataFrame) -> pd.DataFrame:
    lat_col, lon_col = pick_lat_lon_cols(df)
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["lat", "lon"])

    pts = df[[lat_col, lon_col]].copy()
    pts.columns = ["lat", "lon"]
    pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts = pts.dropna(subset=["lat", "lon"])
    pts = pts[(pts["lat"].between(-90, 90)) & (pts["lon"].between(-180, 180))]
    return pts


def points_to_pixels(
    pts: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    width: int,
    height: int,
) -> pd.DataFrame:
    if pts.empty:
        return pd.DataFrame(columns=["px", "py"])
    out = pts.copy()
    out["px"] = (out["lon"] - min_lon) / (max_lon - min_lon) * (width - 1)
    out["py"] = (max_lat - out["lat"]) / (max_lat - min_lat) * (height - 1)
    out["px"] = out["px"].clip(0, width - 1)
    out["py"] = out["py"].clip(0, height - 1)
    return out[["px", "py"]]


def density_grid(pxpy: pd.DataFrame, width: int, height: int, bins: int = GRID_SIZE) -> np.ndarray:
    if pxpy.empty:
        return np.zeros((bins, bins), dtype=float)
    hist, _, _ = np.histogram2d(
        pxpy["py"],
        pxpy["px"],
        bins=[bins, bins],
        range=[[0, height], [0, width]],
    )
    return hist


def circular_smooth_grid(grid: np.ndarray, radius: int = CIRCLE_RADIUS, sigma: float = CIRCLE_SIGMA) -> np.ndarray:
    # Apply a circular Gaussian kernel so hotspots are radial (not square bins).
    r = int(max(1, radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    dist2 = x * x + y * y
    kernel = np.exp(-dist2 / (2.0 * sigma * sigma))
    kernel[dist2 > (r * r)] = 0.0
    ksum = float(kernel.sum())
    if ksum <= 0:
        return grid.astype(float, copy=True)
    kernel /= ksum

    gh, gw = grid.shape
    kh, kw = kernel.shape

    # FFT-based convolution: fast and avoids square artifacts from coarse bins.
    out_h = gh + kh - 1
    out_w = gw + kw - 1
    g_fft = np.fft.rfft2(grid, s=(out_h, out_w))
    k_fft = np.fft.rfft2(kernel, s=(out_h, out_w))
    conv = np.fft.irfft2(g_fft * k_fft, s=(out_h, out_w))

    start_y = kh // 2
    start_x = kw // 2
    return conv[start_y : start_y + gh, start_x : start_x + gw]


def normalize(grid: np.ndarray, clip_q: float = HEAT_CLIP_QUANTILE, gamma: float = HEAT_GAMMA) -> np.ndarray:
    if grid.size == 0:
        return grid
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return grid

    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros_like(grid, dtype=float)

    gmax = float(np.quantile(positive, clip_q))
    if gmax <= 0:
        return np.zeros_like(grid, dtype=float)

    scaled = np.clip(grid / gmax, 0.0, 1.0)
    heat = np.power(scaled, gamma)
    # Hide very low background values so the map is not fully tinted.
    heat[heat < MIN_VISIBLE_FRACTION] = 0.0
    return heat


def sanitize_name(name: str) -> str:
    base = os.path.splitext(name)[0]
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")


def save_overlay(
    map_img: np.ndarray,
    layer: np.ndarray,
    title: str,
    out_name: str,
    clip_q: float = HEAT_CLIP_QUANTILE,
    gamma: float = HEAT_GAMMA,
    alpha: float = 0.70,
    pxpy: pd.DataFrame | None = None,
    show_points: bool = False,
    radius: int = CIRCLE_RADIUS,
    sigma: float = CIRCLE_SIGMA,
) -> None:
    h, w = map_img.shape[0], map_img.shape[1]
    dpi = 200
    # Keep the map at native scale and place title/key in external border panels.
    side_w = 220
    top_h = 80
    total_w = w + side_w
    total_h = h + top_h

    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
    map_left = 0.0
    map_bottom = 0.0
    map_width = w / total_w
    map_height = h / total_h

    ax = fig.add_axes([map_left, map_bottom, map_width, map_height])
    ax.imshow(map_img, extent=[0, w, h, 0], interpolation="nearest")
    heat_img = normalize(circular_smooth_grid(layer, radius=radius, sigma=sigma), clip_q=clip_q, gamma=gamma)
    heat_masked = np.ma.masked_where(heat_img <= 0, heat_img)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(alpha=0.0)
    hm = ax.imshow(
        heat_masked,
        cmap=cmap,
        alpha=alpha,
        extent=[0, w, h, 0],
        interpolation="bicubic",
    )
    if show_points and pxpy is not None and not pxpy.empty:
        ax.scatter(pxpy["px"], pxpy["py"], s=2, c="#00ffff", alpha=0.18, linewidths=0)
    # Top panel title.
    ax_top = fig.add_axes([0.0, h / total_h, w / total_w, top_h / total_h])
    ax_top.set_facecolor("white")
    ax_top.text(0.02, 0.5, title, va="center", ha="left", fontsize=14, color="black")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # Right panel for color key and legend.
    ax_side = fig.add_axes([w / total_w, 0.0, side_w / total_w, 1.0])
    ax_side.set_facecolor("white")
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    for spine in ax_side.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([(w + 70) / total_w, (h * 0.18) / total_h, 30 / total_w, (h * 0.62) / total_h])
    cbar = fig.colorbar(hm, cax=cax)
    cbar.set_label("Relative intensity", rotation=90)

    if show_points and pxpy is not None and not pxpy.empty:
        marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00ffff",
            markersize=6,
            alpha=0.8,
            linewidth=0,
            label="Raw points",
        )
        ax_side.legend(handles=[marker], loc="lower center", frameon=True, fontsize=10)
    ax.axis("off")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_path}")


if not os.path.exists(MAP_IMAGE):
    raise FileNotFoundError(f"Map image not found: {MAP_IMAGE}")

csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError("No CSV files found.")

print("Scanning CSV files for location data...")
datasets: list[tuple[str, pd.DataFrame]] = []
all_points_list: list[pd.DataFrame] = []
crime_2023_points: list[pd.DataFrame] = []
crime_2024_points: list[pd.DataFrame] = []

for path in csv_files:
    name = os.path.basename(path)
    df = standardize_column_names(pd.read_csv(path, low_memory=False))
    pts = extract_points(df)
    if pts.empty:
        print(f"Skipping {name}: no usable lat/lon columns.")
        continue
    if name.startswith("Crime_Data_2023"):
        crime_2023_points.append(pts)
        print(f"Loaded {name} for 2023 combine: {len(pts)} points")
        continue
    if name.startswith("Crime_Data_2024"):
        crime_2024_points.append(pts)
        print(f"Loaded {name} for 2024 combine: {len(pts)} points")
        continue
    datasets.append((name, pts))
    all_points_list.append(pts)
    print(f"Loaded {name}: {len(pts)} points")

if crime_2023_points:
    combined_2023 = pd.concat(crime_2023_points, ignore_index=True)
    datasets.append(("Crime_Data_2023_Combined.csv", combined_2023))
    all_points_list.append(combined_2023)
    print(f"Loaded Crime_Data_2023_Combined.csv: {len(combined_2023)} points")

if crime_2024_points:
    combined_2024 = pd.concat(crime_2024_points, ignore_index=True)
    datasets.append(("Crime_Data_2024_Combined.csv", combined_2024))
    all_points_list.append(combined_2024)
    print(f"Loaded Crime_Data_2024_Combined.csv: {len(combined_2024)} points")

if not datasets:
    raise ValueError("No CSV files with usable location columns were found.")

all_points = pd.concat(all_points_list, ignore_index=True)
if MAP_BOUNDS is None:
    min_lat = float(all_points["lat"].quantile(EXTENT_LOW_Q))
    max_lat = float(all_points["lat"].quantile(EXTENT_HIGH_Q))
    min_lon = float(all_points["lon"].quantile(EXTENT_LOW_Q))
    max_lon = float(all_points["lon"].quantile(EXTENT_HIGH_Q))
else:
    min_lat = float(MAP_BOUNDS["min_lat"])
    max_lat = float(MAP_BOUNDS["max_lat"])
    min_lon = float(MAP_BOUNDS["min_lon"])
    max_lon = float(MAP_BOUNDS["max_lon"])
if min_lat == max_lat or min_lon == max_lon:
    raise ValueError("Spatial extent is invalid (min == max).")

map_img = plt.imread(MAP_IMAGE)
height, width = map_img.shape[0], map_img.shape[1]

print("Generating one heatmap per CSV...")
for name, pts in datasets:
    pxpy = points_to_pixels(pts, min_lat, max_lat, min_lon, max_lon, width, height)
    if name.startswith("Crime_Data_"):
        scale = {
            "clip_q": 0.90,
            "gamma": 0.46,
            "alpha": 0.51,
            "bins": 150,
            "show_points": False,
            "radius": CIRCLE_RADIUS,
            "sigma": CIRCLE_SIGMA,
        }
    else:
        scale = SPECIAL_SCALE.get(
            name,
            {
                "clip_q": HEAT_CLIP_QUANTILE,
                "gamma": HEAT_GAMMA,
                "alpha": 0.70,
                "bins": GRID_SIZE,
                "show_points": False,
                "radius": CIRCLE_RADIUS,
                "sigma": CIRCLE_SIGMA,
            },
        )

    grid = density_grid(pxpy, width, height, bins=scale["bins"])
    out_file = f"heatmap_{sanitize_name(name)}.png"
    save_overlay(
        map_img,
        grid,
        f"Location Heatmap: {name}",
        out_file,
        clip_q=scale["clip_q"],
        gamma=scale["gamma"],
        alpha=scale["alpha"],
        pxpy=pxpy,
        show_points=scale["show_points"],
        radius=scale["radius"],
        sigma=scale["sigma"],
    )

print("Done. Outputs saved in:", OUTPUT_DIR)


