import rasterio
import numpy as np
import xarray as xr

def load_tif(path):

    with rasterio.open(path) as src:

        data = src.read(1)

    return data


def load_rainfall_nc(path):

    ds = xr.open_dataset(path)

    var_name = list(ds.data_vars)[0]

    rainfall = ds[var_name]

    rainfall_mean = rainfall.mean(dim="time")

    return rainfall_mean.values


def resize_to_match(data, target_shape):

    from skimage.transform import resize

    return resize(
        data,
        target_shape,
        preserve_range=True
    )