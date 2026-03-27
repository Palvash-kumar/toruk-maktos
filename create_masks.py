import cv2
import numpy as np
import rasterio
import os

INPUT_FOLDER = "ground_truth_parts"

OUTPUT_FOLDER = "ground_truth_masks"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

files = os.listdir(INPUT_FOLDER)

for file in files:

    path = os.path.join(
        INPUT_FOLDER,
        file
    )

    print("Processing:", file)

    img = cv2.imread(path)

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_yellow = np.array(
        [20, 100, 100]
    )

    upper_yellow = np.array(
        [35, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    mask = (
        mask > 0
    ).astype("uint8")

    height, width = mask.shape

    output_path = os.path.join(
        OUTPUT_FOLDER,
        file.replace(
            ".png",
            "_mask.tif"
        )
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8"
    }

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(mask, 1)

    print("Saved:", output_path)