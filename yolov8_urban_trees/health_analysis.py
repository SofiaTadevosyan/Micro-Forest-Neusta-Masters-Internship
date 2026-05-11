#!/usr/bin/env python3
"""
health_analysis.py

NDVI-based tree health analysis using the NIR band from NAIP aerial imagery.

For each detected tree bounding box, computes the mean NDVI over the canopy
pixels and classifies health status as Healthy / Moderate / Stressed.

NDVI = (NIR - Red) / (NIR + Red)
  - NIR  = channel 3 (index 3) of the 4-band NAIP .npy file
  - Red  = channel 0 (index 0)
  - Range: -1 to +1. Vegetation typically 0.2 – 0.8.

Health thresholds (conservative for young urban plantations):
  > 0.4  → Healthy   (active photosynthesis, good canopy)
  0.2–0.4 → Moderate  (mild stress, shade, or young growth)
  < 0.2  → Stressed  (hydric/disease stress, sparse canopy)

These thresholds are documented as approximate — the previous PIE ETE-03 team
noted there is no consensus on health definitions for young micro-forests.
"""

import numpy as np
import cv2

# ─── NDVI thresholds ────────────────────────────────────────────────────────
# Adjusted for urban aerial 256px tiles where bounding boxes (30×30 px)
# include mixed pixels (canopy + shadow + pavement). Values are lower than
# open-field NDVI due to the urban context and small box size.
NDVI_HEALTHY   = 0.2
NDVI_MODERATE  = 0.0

# ─── Colors (BGR for OpenCV, RGB for display) ───────────────────────────────
COLOR_HEALTHY  = (0,   200,  80)   # green
COLOR_MODERATE = (0,   180, 220)   # yellow-orange → using cyan for visibility
COLOR_STRESSED = (60,   60, 220)   # red (BGR)

COLOR_HEALTHY_RGB  = (0,   200,  80)
COLOR_MODERATE_RGB = (220, 180,   0)
COLOR_STRESSED_RGB = (220,  60,  60)


def load_rgbn(path: str) -> np.ndarray:
    """Load a 4-channel .npy file. Returns float32 array (H, W, 4) in [0, 1]."""
    arr = np.load(path).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def compute_ndvi(rgbn: np.ndarray, boxes: list) -> list:
    """
    Compute mean NDVI for each bounding box.

    Args:
        rgbn : (H, W, 4) float32 array [0, 1] — channels: R G B NIR
        boxes: list of (x1, y1, x2, y2) pixel coordinates

    Returns:
        List of float NDVI values, one per box. Returns 0.0 for empty boxes.
    """
    H, W = rgbn.shape[:2]
    red = rgbn[:, :, 0]
    nir = rgbn[:, :, 3]
    ndvi_map = (nir - red) / (nir + red + 1e-8)

    ndvi_values = []
    for (x1, y1, x2, y2) in boxes:
        x1c = max(0, int(x1))
        y1c = max(0, int(y1))
        x2c = min(W, int(x2))
        y2c = min(H, int(y2))
        if x2c <= x1c or y2c <= y1c:
            ndvi_values.append(0.0)
            continue
        patch = ndvi_map[y1c:y2c, x1c:x2c]
        ndvi_values.append(float(patch.mean()))

    return ndvi_values


def classify_health(ndvi: float) -> tuple:
    """
    Map NDVI value to (label, emoji, color_rgb).

    Returns:
        (label: str, emoji: str, color_rgb: tuple)
    """
    if ndvi > NDVI_HEALTHY:
        return "Healthy",  "🟢", COLOR_HEALTHY_RGB
    elif ndvi > NDVI_MODERATE:
        return "Moderate", "🟡", COLOR_MODERATE_RGB
    else:
        return "Stressed", "🔴", COLOR_STRESSED_RGB


def draw_health_boxes(img_rgb: np.ndarray, boxes: list, ndvi_values: list) -> np.ndarray:
    """
    Draw color-coded bounding boxes on an RGB image.
    Green = Healthy, Yellow = Moderate, Red = Stressed.
    Each box is labelled with its NDVI value.

    Args:
        img_rgb    : (H, W, 3) uint8 RGB image
        boxes      : list of (x1, y1, x2, y2)
        ndvi_values: list of float NDVI values

    Returns:
        Annotated RGB image.
    """
    out = img_rgb.copy()
    # Work in BGR for OpenCV drawing
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for (x1, y1, x2, y2), ndvi in zip(boxes, ndvi_values):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label, _, color_rgb = classify_health(ndvi)
        # BGR for OpenCV
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        # Small NDVI label above box
        txt = f"{ndvi:.2f}"
        cv2.putText(out_bgr, txt, (x1, max(y1 - 3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color_bgr, 1, cv2.LINE_AA)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def health_summary(ndvi_values: list) -> dict:
    """
    Compute aggregate health statistics.

    Returns dict with keys:
        n_healthy, n_moderate, n_stressed,
        pct_healthy, pct_moderate, pct_stressed,
        mean_ndvi, min_ndvi, max_ndvi
    """
    if not ndvi_values:
        return {k: 0 for k in
                ["n_healthy", "n_moderate", "n_stressed",
                 "pct_healthy", "pct_moderate", "pct_stressed",
                 "mean_ndvi", "min_ndvi", "max_ndvi"]}

    n_h = sum(1 for v in ndvi_values if v > NDVI_HEALTHY)
    n_m = sum(1 for v in ndvi_values if NDVI_MODERATE < v <= NDVI_HEALTHY)
    n_s = sum(1 for v in ndvi_values if v <= NDVI_MODERATE)
    total = len(ndvi_values)

    return {
        "n_healthy":   n_h,
        "n_moderate":  n_m,
        "n_stressed":  n_s,
        "pct_healthy":  round(100 * n_h / total),
        "pct_moderate": round(100 * n_m / total),
        "pct_stressed": round(100 * n_s / total),
        "mean_ndvi": round(float(np.mean(ndvi_values)), 3),
        "min_ndvi":  round(float(np.min(ndvi_values)),  3),
        "max_ndvi":  round(float(np.max(ndvi_values)),  3),
    }
