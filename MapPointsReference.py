"""
geo_overlay.py
Opens a live window showing the map, then plots points in animated batches.
"""

import csv
import sys
import time
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk

# ── Bounding box ─────────────────────────────────────────────────────────────
LAT_TOP   =  43.087394
LAT_BOT   =  42.977589
LON_LEFT  = -76.217281
LON_RIGHT = -76.079200

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH  = "your_map.png"
POINTS_PATH = "Vacant_Properties.csv"
BATCH_SIZE  = 5      # how many points to add per frame
INTERVAL_MS = 800    # milliseconds between each batch

# ── Marker style ─────────────────────────────────────────────────────────────
MARKER_RADIUS  = 2
MARKER_COLOR   = (220, 50, 50, 220)
MARKER_OUTLINE = (255, 255, 255, 255)


def latlon_to_pixel(lat, lon, w, h):
    x = (lon - LON_LEFT)  / (LON_RIGHT - LON_LEFT)
    y = (LAT_TOP  - lat)  / (LAT_TOP   - LAT_BOT)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return int(x * w), int(y * h)


def load_points(path):
    points = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                if lat == 0.0 and lon == 0.0:
                    continue  # skip missing/null coordinates
                points.append((lat, lon))
            except (KeyError, ValueError):
                continue
    return points


def draw_marker(draw, cx, cy, r=MARKER_RADIUS):
    draw.ellipse([cx-r-1, cy-r-1, cx+r+1, cy+r+1], fill=MARKER_OUTLINE)
    draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r  ], fill=MARKER_COLOR)


class MapAnimator:
    def __init__(self, root, base_img, points):
        self.root   = root
        self.points = points
        self.index  = 0
        self.w, self.h = base_img.size

        # Keep a persistent RGBA canvas we draw onto
        self.canvas_img = base_img.convert("RGBA")

        # Tkinter canvas
        self.tk_canvas = tk.Canvas(root, width=self.w, height=self.h)
        self.tk_canvas.pack()

        # Status label
        self.label = tk.Label(root, text="Starting...", font=("Arial", 11))
        self.label.pack(pady=4)

        self.photo = None   # must keep reference to avoid GC
        self.render()       # show the base map immediately
        self.root.after(INTERVAL_MS, self.step)

    def render(self):
        """Push the current canvas_img to the Tkinter window."""
        display = self.canvas_img.convert("RGB")
        self.photo = ImageTk.PhotoImage(display)
        self.tk_canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def step(self):
        """Add the next batch of points, then schedule the next step."""
        if self.index >= len(self.points):
            self.label.config(text=f"Done! {len(self.points)} points plotted. You may close the window.")
            self.save_final()
            return

        batch_end = min(self.index + BATCH_SIZE, len(self.points))
        draw = ImageDraw.Draw(self.canvas_img)

        for lat, lon in self.points[self.index:batch_end]:
            px, py = latlon_to_pixel(lat, lon, self.w, self.h)
            draw_marker(draw, px, py)

        self.index = batch_end
        self.label.config(
            text=f"Plotting point {self.index} / {len(self.points)}"
        )
        self.render()
        self.root.after(INTERVAL_MS, self.step)


def main():
    points = load_points(POINTS_PATH)
    if not points:
        sys.exit("No valid points found in CSV.")

    base_img = Image.open(IMAGE_PATH)
    print(f"Loaded map: {base_img.size[0]}×{base_img.size[1]} px")
    print(f"Loaded {len(points)} points — animating in batches of {BATCH_SIZE}")

    root = tk.Tk()
    root.title("Map Overlay — Live Plot")
    root.resizable(False, False)

    MapAnimator(root, base_img, points)
    root.mainloop()


if __name__ == "__main__":
    main()
