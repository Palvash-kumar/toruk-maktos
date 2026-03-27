import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "newdataset"
)

DEM_PATH = os.path.join(
    DATA_DIR,
    "DEM",
    "Copernicus.tif"
)

SENTINEL1_PATH = os.path.join(
    DATA_DIR,
    "Sentinel-1"
)

SENTINEL2_PATH = os.path.join(
    DATA_DIR,
    "Sentinel-2"
)

RAIN_PATH = os.path.join(
    DATA_DIR,
    "Rainfall",
    "rainfall_2019.nc"
)

SOIL_PATH = os.path.join(
    DATA_DIR,
    "soil"
)

INPUT_DIM = 5

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001