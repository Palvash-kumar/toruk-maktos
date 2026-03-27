"""
generate_labels.py
═══════════════════════════════════════════════════════════════════════════════
Heuristic proxy-label generator for landslide susceptibility.

Until binary rasters are manually digitised from LandslideAtlas_new_2023.pdf,
this module produces approximate landslide / non-landslide labels by combining
multi-source evidence:

    1. DEM Slope     — steep slopes (>25°) increase susceptibility
    2. NDVI anomaly  — bare soil / scar zones have low NDVI (<0.2)
    3. Soil Moisture — saturated soil (high values) triggers slides
    4. Rainfall      — extreme rainfall (high values) triggers slides

Each factor produces a [0,1] susceptibility score; the scores are averaged
and thresholded to yield a binary label map:
    1 = likely landslide zone
    0 = non-landslide zone

Usage:
    from data.generate_labels import generate_heuristic_labels
    label_map = generate_heuristic_labels(site_dir, target_size=(256,256))

NOTE: These are PROXY labels for development. Replace with ground-truth
      rasters extracted from the Landslide Atlas for final results.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import numpy as np
from typing import Tuple, Optional

# ── optional imports (same pattern as data_pipeline.py) ────────────────────────
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _read_band(path: str, band: int = 1) -> np.ndarray:
    """Read one band from a GeoTIFF, replacing nodata with NaN."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
    return arr


def _resize2d(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to target shape."""
    try:
        from scipy.ndimage import zoom
        zy = shape[0] / arr.shape[0]
        zx = shape[1] / arr.shape[1]
        return zoom(arr, (zy, zx), order=1).astype(np.float32)
    except ImportError:
        r = np.linspace(0, arr.shape[0] - 1, shape[0]).astype(int)
        c = np.linspace(0, arr.shape[1] - 1, shape[1]).astype(int)
        r = r.clip(0, arr.shape[0] - 1)
        c = c.clip(0, arr.shape[1] - 1)
        return arr[np.ix_(r, c)].astype(np.float32)


def _normalise_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], ignoring NaN."""
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR 1 — SLOPE FROM DEM
# ══════════════════════════════════════════════════════════════════════════════

def _slope_score(dem_path: str, target_size: Tuple[int, int],
                 slope_threshold_deg: float = 25.0) -> np.ndarray:
    """
    Compute slope from DEM via np.gradient, then score 0→1.
    Slopes > threshold get score → 1.
    """
    dem = _read_band(dem_path)
    dem = _resize2d(dem, target_size)
    dem = np.nan_to_num(dem, nan=0.0)

    # Gradient → slope in degrees
    dy, dx = np.gradient(dem)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_deg = np.degrees(slope_rad)

    # Sigmoid-like score: ramps from 0 to 1 around threshold
    score = 1.0 / (1.0 + np.exp(-(slope_deg - slope_threshold_deg) / 5.0))
    return score.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR 2 — LOW NDVI (bare soil / landslide scar)
# ══════════════════════════════════════════════════════════════════════════════

def _ndvi_score(sentinel2_dir: str, target_size: Tuple[int, int],
                ndvi_threshold: float = 0.2) -> np.ndarray:
    """
    Low NDVI → high susceptibility. Score inverted:
    NDVI < threshold → score ≈ 1 (bare/scar), NDVI > 0.5 → score ≈ 0 (vegetation).
    """
    b08_path = os.path.join(sentinel2_dir, 'B08.tif')
    b04_path = os.path.join(sentinel2_dir, 'B04.tif')

    if not (os.path.exists(b08_path) and os.path.exists(b04_path)):
        return np.full(target_size, 0.5, dtype=np.float32)

    b08 = _resize2d(_read_band(b08_path), target_size)
    b04 = _resize2d(_read_band(b04_path), target_size)

    ndvi = (b08 - b04) / (b08 + b04 + 1e-8)
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Invert: low NDVI → high score
    score = 1.0 / (1.0 + np.exp((ndvi - ndvi_threshold) / 0.05))
    return score.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR 3 — HIGH SOIL MOISTURE
# ══════════════════════════════════════════════════════════════════════════════

def _soil_moisture_score(sm_dir: str, event_date: str,
                         target_size: Tuple[int, int]) -> np.ndarray:
    """
    High soil moisture → high susceptibility.
    Picks SMAP .tif closest to event_date.
    """
    tifs = sorted(glob.glob(os.path.join(sm_dir, 'SM_SMAP_I_*.tif')))
    if not tifs:
        return np.full(target_size, 0.5, dtype=np.float32)

    def date_dist(fp):
        fn = os.path.basename(fp)
        parts = fn.replace('SM_SMAP_I_', '').replace('.tif', '').split('_')
        return abs(int(parts[0]) - int(event_date))

    closest = min(tifs, key=date_dist)
    sm = _read_band(closest)
    sm = _resize2d(sm, target_size)
    sm = np.nan_to_num(sm, nan=0.0)

    return _normalise_01(sm)


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR 4 — HIGH RAINFALL
# ══════════════════════════════════════════════════════════════════════════════

def _rainfall_score(rainfall_path: str,
                    target_size: Tuple[int, int]) -> np.ndarray:
    """
    High rainfall → high susceptibility.
    Loads NetCDF, takes max over time, normalises to [0,1].
    """
    if not rainfall_path or not os.path.exists(rainfall_path):
        return np.full(target_size, 0.5, dtype=np.float32)

    if not HAS_NC:
        return np.full(target_size, 0.5, dtype=np.float32)

    with nc.Dataset(rainfall_path) as ds:
        coords = {'lat', 'lon', 'latitude', 'longitude', 'time', 'level'}
        for vname in ['RAINFALL', 'rainfall', 'precip', 'pr', 'rain']:
            if vname in ds.variables:
                rain = ds.variables[vname][:]
                break
        else:
            vname = next(v for v in ds.variables if v.lower() not in coords)
            rain = ds.variables[vname][:]

    rain = np.array(rain, dtype=np.float32)
    rain = np.ma.filled(rain, np.nan) if hasattr(rain, 'filled') else rain

    if rain.ndim == 3:
        rain = np.nanmax(rain, axis=0)

    rain = _resize2d(rain, target_size)
    rain = np.nan_to_num(rain, nan=0.0)
    return _normalise_01(rain)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE LABEL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_heuristic_labels(
    dem_path:       str,
    sentinel2_dir:  str,
    sm_dir:         str,
    event_date:     str,
    rainfall_path:  str = '',
    target_size:    Tuple[int, int] = (256, 256),
    threshold:      float = 0.55,
    weights:        Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """
    Generate binary heuristic labels by combining multiple susceptibility
    factors.

    Args:
        dem_path:      Path to Copernicus DEM GeoTIFF
        sentinel2_dir: Path to Sentinel-2 band directory
        sm_dir:        Path to SMAP soil moisture directory
        event_date:    'YYYYMMDD' — event date for SM selection
        rainfall_path: Path to rainfall .nc file (optional)
        target_size:   (H, W) output size
        threshold:     Composite score threshold for label=1
        weights:       (w_slope, w_ndvi, w_sm, w_rain) — default (0.35, 0.30, 0.20, 0.15)

    Returns:
        (H, W) uint8 array — 1=landslide, 0=non-landslide
    """
    if weights is None:
        weights = (0.35, 0.30, 0.20, 0.15)

    if not HAS_RASTERIO:
        print("[WARNING] rasterio not installed — returning uniform random labels")
        rng = np.random.RandomState(42)
        return (rng.rand(*target_size) > 0.7).astype(np.uint8)

    # Compute each factor
    scores = []
    names = []

    if os.path.exists(dem_path):
        scores.append(_slope_score(dem_path, target_size))
        names.append('slope')
    else:
        scores.append(np.full(target_size, 0.3, dtype=np.float32))
        names.append('slope(missing)')

    if os.path.isdir(sentinel2_dir):
        scores.append(_ndvi_score(sentinel2_dir, target_size))
        names.append('ndvi')
    else:
        scores.append(np.full(target_size, 0.3, dtype=np.float32))
        names.append('ndvi(missing)')

    if os.path.isdir(sm_dir):
        scores.append(_soil_moisture_score(sm_dir, event_date, target_size))
        names.append('soil_moisture')
    else:
        scores.append(np.full(target_size, 0.3, dtype=np.float32))
        names.append('sm(missing)')

    scores.append(_rainfall_score(rainfall_path, target_size))
    names.append('rainfall')

    # Weighted average
    w = np.array(weights[:len(scores)], dtype=np.float32)
    w = w / w.sum()

    composite = np.zeros(target_size, dtype=np.float32)
    for s, wt, name in zip(scores, w, names):
        composite += wt * s

    # Threshold → binary
    label_map = (composite >= threshold).astype(np.uint8)

    pct = label_map.mean() * 100
    print(f"  [Heuristic Labels] {pct:.1f}% pixels labelled as landslide "
          f"(threshold={threshold:.2f})")
    print(f"  [Heuristic Labels] Factors: {', '.join(names)}")
    print(f"  [Heuristic Labels] Weights: {dict(zip(names, w.tolist()))}")

    return label_map


# ══════════════════════════════════════════════════════════════════════════════
# SITE-SPECIFIC WRAPPERS (matching exact folder layout)
# ══════════════════════════════════════════════════════════════════════════════

def generate_puthumala_labels(
    dataset_root: str,
    target_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.55,
) -> np.ndarray:
    """Generate heuristic labels for Puthumala training site."""
    base = os.path.join(dataset_root, 'Puthumala-Training_data')
    return generate_heuristic_labels(
        dem_path      = os.path.join(base, 'DEM', 'Copernicus_DEM_30m.tif'),
        sentinel2_dir = os.path.join(base, 'Sentinel-2'),
        sm_dir        = os.path.join(base, 'Soil_Mositure', '2019'),
        event_date    = '20190807',
        rainfall_path = os.path.join(base, 'Rainfall',
                                     'imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc'),
        target_size   = target_size,
        threshold     = threshold,
    )


def generate_wayanad_labels(
    dataset_root: str,
    date_folder: str = '2024-12-11',
    target_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.55,
) -> np.ndarray:
    """Generate heuristic labels for Wayanad validation site."""
    base = os.path.join(dataset_root, 'Wayanad_validation_data', date_folder)

    # Rainfall folder has a space
    rain_dir = os.path.join(base, 'Rainfall Data')
    rain_path = ''
    if os.path.isdir(rain_dir):
        ncs = glob.glob(os.path.join(rain_dir, '*.nc'))
        rain_path = ncs[0] if ncs else ''

    # Nested soil moisture path (handles the typo)
    sm_dir = os.path.join(base, 'Soil_moisture', 'Soil_Mositure')
    if not os.path.isdir(sm_dir):
        sm_dir = os.path.join(base, 'Soil_Mositure')

    return generate_heuristic_labels(
        dem_path      = os.path.join(base, 'DEM', 'Copernicus_DEM_30m.tif'),
        sentinel2_dir = os.path.join(base, 'Sentinel-2'),
        sm_dir        = sm_dir,
        event_date    = '20240730',
        rainfall_path = rain_path,
        target_size   = target_size,
        threshold     = threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD OR GENERATE  (main entry point for train.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate_labels(
    dataset_root: str,
    site: str,                             # 'puthumala' or 'wayanad'
    target_size: Tuple[int, int] = (256, 256),
    label_tif_path: Optional[str] = None,  # path to manually created .tif
    threshold: float = 0.55,
) -> np.ndarray:
    """
    Try to load a manually-created label .tif first.
    Fall back to heuristic generation if not available.

    Search order:
      1. Explicit label_tif_path argument
      2. ground_truth/label_{site}.tif
      3. Heuristic generation from satellite data
    """
    project_root = os.path.dirname(dataset_root)

    # 1) Explicit path
    if label_tif_path and os.path.exists(label_tif_path):
        print(f"  Loading ground-truth labels from {label_tif_path}")
        lbl = _read_band(label_tif_path)
        lbl = _resize2d(lbl, target_size)
        return (lbl > 0.5).astype(np.uint8)

    # 2) Convention path: ground_truth/label_{site}.tif
    conv_path = os.path.join(project_root, 'ground_truth', f'label_{site}.tif')
    if os.path.exists(conv_path):
        print(f"  Loading ground-truth labels from {conv_path}")
        lbl = _read_band(conv_path)
        lbl = _resize2d(lbl, target_size)
        return (lbl > 0.5).astype(np.uint8)

    # 3) Heuristic fallback
    print(f"  [INFO] No ground-truth .tif found for '{site}' — generating heuristic labels")
    print(f"         Place label_{site}.tif in ground_truth/ to use real labels.")

    if site == 'puthumala':
        return generate_puthumala_labels(dataset_root, target_size, threshold)
    elif site == 'wayanad':
        return generate_wayanad_labels(dataset_root, target_size=target_size,
                                       threshold=threshold)
    else:
        raise ValueError(f"Unknown site: {site}. Use 'puthumala' or 'wayanad'.")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser('Generate heuristic landslide labels')
    p.add_argument('--dataset_root', default='dataset')
    p.add_argument('--site', default='puthumala', choices=['puthumala', 'wayanad'])
    p.add_argument('--target_h', type=int, default=256)
    p.add_argument('--target_w', type=int, default=256)
    p.add_argument('--threshold', type=float, default=0.55)
    p.add_argument('--save', default='', help='Save label map as .npy')
    args = p.parse_args()

    labels = load_or_generate_labels(
        args.dataset_root, args.site,
        target_size=(args.target_h, args.target_w),
        threshold=args.threshold,
    )
    print(f"\nLabel map shape: {labels.shape}")
    print(f"Landslide pixels: {labels.sum()} / {labels.size} "
          f"({labels.mean()*100:.1f}%)")

    if args.save:
        np.save(args.save, labels)
        print(f"Saved to {args.save}")
