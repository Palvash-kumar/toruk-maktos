"""
data_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Geospatial data loading and preprocessing for LandslideEEGMoE.

Reads EXACTLY from the folder layout shown in the project tree:

  dataset/
    Puthumala-Training_data/
      DEM/Copernicus_DEM_30m.tif
      Rainfall/imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc
      Sentinel-1/Sentinel-1_2019-12-08_SAR.tif
      Sentinel-2/B02.tif B03.tif B04.tif B08.tif B8A.tif B11.tif B12.tif
      Soil_Mositure/2019/SM_SMAP_I_*.tif          ← note: "Mositure" typo in data

    Wayanad_validation_data/
      2024-12-11/
        DEM/Copernicus_DEM_30m.tif
        Rainfall Data/kerala_rainfall_data.nc      ← folder has space
        Sentinel-1/Sentinel-1_2024-12-11_SAR.tif
        Sentinel-2/B02.tif B03.tif B04.tif B08.tif ...
        Soil_moisture/Soil_Mositure/SM_SMAP_I_*.tif
      2024-12-16/
        DEM/  Sentinel-1/  Sentinel-2/

Feature channel mapping (analogue of EEGMoE 5 frequency bands δ θ α β γ):
  ch 0 : SAR backscatter (Sentinel-1, first band)
  ch 1 : NDVI  = (B08 - B04) / (B08 + B04)
  ch 2 : NDWI  = (B03 - B08) / (B03 + B08)
  ch 3 : Rainfall  (max daily over monsoon window)
  ch 4 : Soil moisture (SMAP, closest file to event date)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader

# ── optional geospatial imports ────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARNING] rasterio not found — install with: pip install rasterio")

try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False
    print("[WARNING] netCDF4 not found — install with: pip install netCDF4")


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL RASTER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def read_tif(path: str, band: int = 1) -> Tuple[np.ndarray, dict]:
    """Read one band from a GeoTIFF. Returns (array, profile)."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required — pip install rasterio")
    with rasterio.open(path) as src:
        arr  = src.read(band).astype(np.float32)
        meta = src.profile.copy()
        meta['transform'] = src.transform
        meta['crs']       = src.crs
        meta['shape']     = (src.height, src.width)
    arr[arr == meta.get('nodata', -9999)] = np.nan
    return arr, meta


def resample_to_match(src_path: str, target_shape: Tuple[int,int],
                      target_transform, target_crs,
                      band: int = 1) -> np.ndarray:
    """Resample a raster to match target spatial extent and resolution."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")
    with rasterio.open(src_path) as src:
        data = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source      = rasterio.band(src, band),
            destination = data,
            src_transform    = src.transform,
            src_crs          = src.crs,
            dst_transform    = target_transform,
            dst_crs          = target_crs,
            resampling       = Resampling.bilinear,
        )
    return data


def zscore(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalise per channel.
    Paper §IV-A-2: "z-score normalization to eliminate abrupt changes"
    """
    mean = np.nanmean(arr)
    std  = np.nanstd(arr)
    return (arr - mean) / (std + eps)


def safe_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 after normalisation."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# INDEX COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def ndvi(b08: np.ndarray, b04: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)  →  vegetation health"""
    return (b08 - b04) / (b08 + b04 + eps)

def ndwi(b03: np.ndarray, b08: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR)  →  surface water content"""
    return (b03 - b08) / (b03 + b08 + eps)


# ══════════════════════════════════════════════════════════════════════════════
# RAINFALL LOADER
# Puthumala : imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc
# Wayanad   : kerala_rainfall_data.nc
# ══════════════════════════════════════════════════════════════════════════════

def load_rainfall_nc(nc_path: str,
                     target_shape: Tuple[int,int]) -> np.ndarray:
    """
    Load rainfall .nc, take max over time dimension,
    then resize to target_shape via bilinear interpolation.
    """
    if not HAS_NC:
        raise RuntimeError("netCDF4 required — pip install netCDF4")
    with nc.Dataset(nc_path) as ds:
        # try common variable names
        for vname in ['RAINFALL', 'rainfall', 'precip', 'pr', 'rain']:
            if vname in ds.variables:
                rain = ds.variables[vname][:]   # (time, lat, lon) or (lat,lon)
                break
        else:
            # take first non-coordinate variable
            coords = {'lat','lon','latitude','longitude','time','level'}
            vname  = next(v for v in ds.variables if v.lower() not in coords)
            rain   = ds.variables[vname][:]

    rain = np.array(rain, dtype=np.float32)
    rain = np.ma.filled(rain, np.nan)

    # collapse time dim → max (most extreme rainfall = highest risk)
    if rain.ndim == 3:
        rain = np.nanmax(rain, axis=0)   # (lat, lon)

    # simple bilinear resize to match Sentinel-2 shape
    rain = _resize2d(rain, target_shape)
    return rain


def _resize2d(arr: np.ndarray, target_shape: Tuple[int,int]) -> np.ndarray:
    """Resize 2D array to target_shape using scipy zoom."""
    try:
        from scipy.ndimage import zoom
        zy = target_shape[0] / arr.shape[0]
        zx = target_shape[1] / arr.shape[1]
        return zoom(arr, (zy, zx), order=1).astype(np.float32)
    except ImportError:
        # fallback: crude nearest-neighbour
        from numpy import interp
        r = np.linspace(0, arr.shape[0]-1, target_shape[0]).astype(int).clip(0, arr.shape[0]-1)
        c = np.linspace(0, arr.shape[1]-1, target_shape[1]).astype(int).clip(0, arr.shape[1]-1)
        return arr[np.ix_(r, c)].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SOIL MOISTURE LOADER  (SMAP .tif files)
# Picks the .tif file whose date is closest to the event date.
# ══════════════════════════════════════════════════════════════════════════════

def load_soil_moisture(sm_dir: str,
                       event_date: str,         # 'YYYYMMDD'
                       target_shape: Tuple[int,int],
                       target_transform=None,
                       target_crs=None) -> np.ndarray:
    """
    Loads the SMAP soil moisture .tif closest to event_date.
    File pattern: SM_SMAP_I_YYYYMMDD_YYYYMMDD.tif
    """
    tifs = sorted(glob.glob(os.path.join(sm_dir, 'SM_SMAP_I_*.tif')))
    if not tifs:
        print(f"[WARNING] No SMAP files in {sm_dir}, returning zeros")
        return np.zeros(target_shape, dtype=np.float32)

    def date_dist(fp):
        fn   = os.path.basename(fp)
        parts = fn.replace('SM_SMAP_I_','').replace('.tif','').split('_')
        d1   = parts[0]   # start date YYYYMMDD
        return abs(int(d1) - int(event_date))

    closest = min(tifs, key=date_dist)

    if target_transform is not None and target_crs is not None and HAS_RASTERIO:
        sm = resample_to_match(closest, target_shape, target_transform, target_crs)
    else:
        sm, _ = read_tif(closest)
        sm    = _resize2d(sm, target_shape)

    return sm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE STACK BUILDER
# Produces the 5-channel (C,H,W) array — analogue of EEGMoE 4D input
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_stack(
    sentinel1_path: str,
    sentinel2_dir:  str,
    rainfall_path:  str,
    soil_moist_dir: str,
    event_date:     str,              # 'YYYYMMDD' e.g. '20190807'
    target_size:    Tuple[int,int] = (256, 256),
) -> np.ndarray:
    """
    Builds the 5-channel feature stack:
      ch0 : SAR backscatter (Sentinel-1, band 1)
      ch1 : NDVI  (B08, B04)
      ch2 : NDWI  (B03, B08)
      ch3 : Rainfall (max over period)
      ch4 : Soil moisture (SMAP, closest to event)

    Returns: (5, H, W) float32, z-score normalised per channel.
    """
    H, W = target_size

    # ── Sentinel-1 SAR ────────────────────────────────────────────────────
    sar, meta = read_tif(sentinel1_path, band=1)
    sar = _resize2d(sar, (H, W))
    t_transform = meta['transform']
    t_crs       = meta['crs']

    # ── Sentinel-2 bands ──────────────────────────────────────────────────
    def s2(band_name):
        p = os.path.join(sentinel2_dir, f'{band_name}.tif')
        if not os.path.exists(p):
            print(f"[WARNING] {p} missing, using zeros")
            return np.zeros((H, W), dtype=np.float32)
        arr, _ = read_tif(p)
        return _resize2d(arr.astype(np.float32), (H, W))

    b03 = s2('B03')   # Green
    b04 = s2('B04')   # Red
    b08 = s2('B08')   # NIR

    nv = ndvi(b08, b04)
    nw = ndwi(b03, b08)

    # ── Rainfall ──────────────────────────────────────────────────────────
    if os.path.exists(rainfall_path):
        rain = load_rainfall_nc(rainfall_path, (H, W))
    else:
        print(f"[WARNING] Rainfall not found: {rainfall_path}")
        rain = np.zeros((H, W), dtype=np.float32)

    # ── Soil moisture ─────────────────────────────────────────────────────
    sm = load_soil_moisture(soil_moist_dir, event_date, (H, W))

    # ── Stack + z-score  (§IV-A-2 analogue) ──────────────────────────────
    stack = np.stack([sar, nv, nw, rain, sm], axis=0).astype(np.float32)
    for c in range(stack.shape[0]):
        stack[c] = safe_nan(zscore(stack[c]))

    return stack   # (5, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET-SPECIFIC LOADERS
# Matching the exact folder structure from the project tree
# ══════════════════════════════════════════════════════════════════════════════

def load_puthumala(dataset_root: str,
                   target_size: Tuple[int,int] = (256, 256)) -> np.ndarray:
    """
    Load Puthumala-Training_data feature stack.
    Event date: 2019-08-07  (Puthumala landslide)
    """
    base = os.path.join(dataset_root, 'Puthumala-Training_data')
    return build_feature_stack(
        sentinel1_path = os.path.join(base, 'Sentinel-1', 'Sentinel-1_2019-12-08_SAR.tif'),
        sentinel2_dir  = os.path.join(base, 'Sentinel-2'),
        rainfall_path  = os.path.join(base, 'Rainfall',
                                       'imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc'),
        soil_moist_dir = os.path.join(base, 'Soil_Mositure', '2019'),
        event_date     = '20190807',
        target_size    = target_size,
    )


def load_wayanad(dataset_root: str,
                 date_folder:  str = '2024-12-11',
                 target_size:  Tuple[int,int] = (256, 256)) -> np.ndarray:
    """
    Load Wayanad_validation_data feature stack.
    Event date: 2024-07-30  (Wayanad landslide)
    date_folder: '2024-12-11' or '2024-12-16'
    """
    base = os.path.join(dataset_root, 'Wayanad_validation_data', date_folder)

    # Rainfall folder has a space in the name
    rain_dir  = os.path.join(base, 'Rainfall Data')
    rain_path = ''
    if os.path.isdir(rain_dir):
        ncs = glob.glob(os.path.join(rain_dir, '*.nc'))
        rain_path = ncs[0] if ncs else ''

    # Soil moisture: nested Soil_moisture/Soil_Mositure/
    sm_dir = os.path.join(base, 'Soil_moisture', 'Soil_Mositure')
    if not os.path.isdir(sm_dir):
        sm_dir = os.path.join(base, 'Soil_Mositure')   # fallback

    sar_date = date_folder.replace('-', '')             # '20241211'
    return build_feature_stack(
        sentinel1_path = os.path.join(base, 'Sentinel-1',
                                       f'Sentinel-1_{date_folder}_SAR.tif'),
        sentinel2_dir  = os.path.join(base, 'Sentinel-2'),
        rainfall_path  = rain_path,
        soil_moist_dir = sm_dir,
        event_date     = '20240730',
        target_size    = target_size,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PATCH EXTRACTION  (analogue of EEGMoE time-window segmentation §IV-A-2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_patches(feature_map: np.ndarray,
                    label_map:   np.ndarray,
                    patch_size:  int = 64,
                    stride:      int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over (C,H,W) feature map and binary (H,W) label map.
    label_map: 1 = landslide, 0 = non-landslide  (from Landslide Atlas)
    Returns:
        patches : (N, C, patch_size, patch_size)
        labels  : (N,) int64
    """
    C, H, W = feature_map.shape
    patches, labels = [], []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = feature_map[:, r:r+patch_size, c:c+patch_size]
            lp    = label_map[r:r+patch_size, c:c+patch_size]

            # skip if >50 % is NaN/inf
            if not np.isfinite(patch).mean() > 0.5:
                continue

            patch  = np.nan_to_num(patch, nan=0.0)
            label  = int(lp.mean() >= 0.5)     # majority vote
            patches.append(patch)
            labels.append(label)

    if not patches:
        return (np.empty((0, C, patch_size, patch_size), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    return (np.stack(patches).astype(np.float32),
            np.array(labels, dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class LandslideDataset(Dataset):
    """Labelled dataset for Stage 2 fine-tuning."""
    def __init__(self, patches: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        self.patches = torch.from_numpy(patches.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment

    def __len__(self):  return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx], self.labels[idx]
        if self.augment:
            if torch.rand(1) > 0.5: x = torch.flip(x, [-1])
            if torch.rand(1) > 0.5: x = torch.flip(x, [-2])
        return x, y


class LandslidePretrainDataset(Dataset):
    """Unlabelled dataset for Stage 1 pre-training (no labels needed)."""
    def __init__(self, patches: np.ndarray):
        self.patches = torch.from_numpy(patches.astype(np.float32))

    def __len__(self):              return len(self.patches)
    def __getitem__(self, idx):     return self.patches[idx]


# ══════════════════════════════════════════════════════════════════════════════
# DATA-LOADER FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def get_pretrain_loader(patches: np.ndarray, batch_size: int = 64,
                        num_workers: int = 0) -> DataLoader:
    return DataLoader(LandslidePretrainDataset(patches),
                      batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=False, drop_last=True)


def get_finetune_loaders(patches: np.ndarray, labels: np.ndarray,
                         val_ratio:  float = 0.1,
                         test_ratio: float = 0.1,
                         batch_size: int   = 128,
                         num_workers: int  = 0,
                         seed: int = 42):
    rng  = np.random.RandomState(seed)
    N    = len(patches)
    idx  = rng.permutation(N)
    nt   = int(N * test_ratio)
    nv   = int(N * val_ratio)
    ti, vi, tri = idx[:nt], idx[nt:nt+nv], idx[nt+nv:]

    kw = dict(num_workers=num_workers, pin_memory=False)
    trl = DataLoader(LandslideDataset(patches[tri], labels[tri], augment=True),
                     batch_size=batch_size, shuffle=True,  **kw)
    vl  = DataLoader(LandslideDataset(patches[vi],  labels[vi]),
                     batch_size=batch_size, shuffle=False, **kw)
    tel = DataLoader(LandslideDataset(patches[ti],  labels[ti]),
                     batch_size=batch_size, shuffle=False, **kw)
    return trl, vl, tel


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE  (called from train.py)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_datasets(dataset_root:  str,
                     label_map_put: np.ndarray,   # (H,W) binary from Atlas
                     label_map_way: np.ndarray,   # (H,W) binary from Atlas
                     patch_size:    int = 64,
                     stride:        int = 32,
                     target_size:   Tuple[int,int] = (256, 256)):
    """
    Full data preparation pipeline:
      1. Load real GeoTIFF / NetCDF files from disk
      2. Build 5-channel feature stacks
      3. Extract patches + labels
      4. Return concatenated arrays ready for DataLoaders

    label_map_put / label_map_way: binary arrays (H,W), 1=landslide.
    Extract these from the Landslide Atlas PDF/raster externally.
    """
    print("Loading Puthumala feature stack...")
    put_features = load_puthumala(dataset_root, target_size)
    put_patches, put_labels = extract_patches(put_features, label_map_put,
                                              patch_size, stride)
    print(f"  Puthumala: {put_patches.shape[0]} patches")

    print("Loading Wayanad feature stack (2024-12-11)...")
    way_features = load_wayanad(dataset_root, '2024-12-11', target_size)
    way_patches, way_labels = extract_patches(way_features, label_map_way,
                                              patch_size, stride)
    print(f"  Wayanad:   {way_patches.shape[0]} patches")

    # Concatenate both events for training
    all_patches = np.concatenate([put_patches, way_patches], axis=0)
    all_labels  = np.concatenate([put_labels,  way_labels],  axis=0)
    print(f"  Total:     {all_patches.shape[0]} patches  "
          f"(landslide={all_labels.sum()}, non={len(all_labels)-all_labels.sum()})")

    return all_patches, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK  (for testing without real data)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n: int = 500, C: int = 5, p: int = 64, seed: int = 42):
    rng     = np.random.RandomState(seed)
    patches = rng.randn(n, C, p, p).astype(np.float32)
    labels  = rng.randint(0, 2, n).astype(np.int64)
    return patches, labels
