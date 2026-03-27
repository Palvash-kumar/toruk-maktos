import os
import numpy as np
import rasterio
import xarray as xr

from skimage.transform import resize

import config


# -----------------------------
# CLEAN FUNCTION
# -----------------------------

def clean_array(x, name=""):

    print("Cleaning:", name)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    x = np.clip(
        x,
        -1e6,
        1e6
    )

    return x


# -----------------------------
# SAFE NORMALIZATION
# -----------------------------

def safe_normalize(x):

    x = np.nan_to_num(x)

    mean = np.mean(x)

    std = np.std(x)

    if std == 0:

        std = 1

    return (x - mean) / std


# -----------------------------
# BASIC LOADERS
# -----------------------------

def load_tif(path):

    with rasterio.open(path) as src:

        data = src.read(1)

    data = clean_array(
        data,
        "tif"
    )

    return data


def resize_to_dem(data, dem_shape):

    data = resize(
        data,
        dem_shape,
        preserve_range=True,
        anti_aliasing=True
    )

    data = clean_array(
        data,
        "resize"
    )

    return data


# -----------------------------
# DEM
# -----------------------------

def load_dem():

    print("Loading DEM")

    dem = load_tif(
        config.DEM_PATH
    )

    dem = safe_normalize(dem)

    print(
        "DEM shape:",
        dem.shape
    )

    return dem


# -----------------------------
# SENTINEL-1
# -----------------------------

def load_sentinel1(dem_shape):

    print("Loading Sentinel-1")

    files = sorted([
        f for f in os.listdir(
            config.SENTINEL1_PATH
        )
        if f.endswith(".tif")
    ])

    stack = []

    for file in files:

        path = os.path.join(
            config.SENTINEL1_PATH,
            file
        )

        data = load_tif(path)

        data = resize_to_dem(
            data,
            dem_shape
        )

        data = safe_normalize(data)

        stack.append(data)

    stack = np.stack(
        stack,
        axis=0
    )

    sentinel1 = stack.mean(
        axis=0
    )

    sentinel1 = clean_array(
        sentinel1,
        "Sentinel-1"
    )

    print(
        "Sentinel-1 shape:",
        sentinel1.shape
    )

    return sentinel1


# -----------------------------
# SENTINEL-2
# -----------------------------

def load_sentinel2(dem_shape):

    print("Loading Sentinel-2")

    files = sorted([
        f for f in os.listdir(
            config.SENTINEL2_PATH
        )
        if f.endswith(".tif")
    ])

    bands = []

    for file in files:

        path = os.path.join(
            config.SENTINEL2_PATH,
            file
        )

        data = load_tif(path)

        data = resize_to_dem(
            data,
            dem_shape
        )

        data = safe_normalize(data)

        bands.append(data)

    sentinel2 = np.stack(
        bands,
        axis=-1
    )

    sentinel2 = clean_array(
        sentinel2,
        "Sentinel-2"
    )

    print(
        "Sentinel-2 shape:",
        sentinel2.shape
    )

    return sentinel2


# -----------------------------
# RAINFALL
# -----------------------------

def load_rainfall(dem_shape):

    print("Loading Rainfall")

    ds = xr.open_dataset(
        config.RAIN_PATH
    )

    var_name = list(
        ds.data_vars
    )[0]

    rainfall = ds[var_name]

    rainfall_mean = rainfall.mean(
        dim="time"
    )

    rainfall_array = rainfall_mean.values

    rainfall_array = resize_to_dem(
        rainfall_array,
        dem_shape
    )

    rainfall_array = safe_normalize(
        rainfall_array
    )

    rainfall_array = clean_array(
        rainfall_array,
        "Rainfall"
    )

    print(
        "Rainfall shape:",
        rainfall_array.shape
    )

    return rainfall_array


# -----------------------------
# SOIL
# -----------------------------

def load_soil(dem_shape):

    print("Loading Soil")

    files = sorted([
        f for f in os.listdir(
            config.SOIL_PATH
        )
        if f.endswith(".tif")
    ])

    stack = []

    for file in files:

        path = os.path.join(
            config.SOIL_PATH,
            file
        )

        data = load_tif(path)

        data = resize_to_dem(
            data,
            dem_shape
        )

        data = safe_normalize(data)

        stack.append(data)

    stack = np.stack(
        stack,
        axis=0
    )

    soil = stack.mean(
        axis=0
    )

    soil = clean_array(
        soil,
        "Soil"
    )

    print(
        "Soil shape:",
        soil.shape
    )

    return soil


# -----------------------------
# LOAD ALL DATA
# -----------------------------

def load_all_data():

    dem = load_dem()

    sentinel1 = load_sentinel1(
        dem.shape
    )

    sentinel2 = load_sentinel2(
        dem.shape
    )

    rainfall = load_rainfall(
        dem.shape
    )

    soil = load_soil(
        dem.shape
    )

    return (
        dem,
        sentinel1,
        sentinel2,
        rainfall,
        soil
    )


# -----------------------------
# CREATE FEATURES
# -----------------------------

def create_features(
    sentinel1,
    sentinel2,
    rainfall,
    soil
):

    print("Creating features")

    features = np.stack(
        [
            sentinel1,
            sentinel2[:, :, 0],
            sentinel2[:, :, 1],
            rainfall,
            soil
        ],
        axis=-1
    )

    features = clean_array(
        features,
        "Features"
    )

    print(
        "Feature shape:",
        features.shape
    )

    return features


# -----------------------------
# CREATE LABELS
# -----------------------------

def create_labels(
    rainfall,
    soil,
    dem
):

    print("Generating pseudo labels")

    dx, dy = np.gradient(dem)

    slope = np.sqrt(
        dx**2 + dy**2
    )

    rainfall_thr = np.percentile(
        rainfall,
        75
    )

    soil_thr = np.percentile(
        soil,
        75
    )

    slope_thr = np.percentile(
        slope,
        75
    )

    labels = (
        (rainfall > rainfall_thr)
        |
        (soil > soil_thr)
        |
        (slope > slope_thr)
    )

    labels = labels.astype(int)

    print(
        "Unique labels:",
        np.unique(labels)
    )

    return labels