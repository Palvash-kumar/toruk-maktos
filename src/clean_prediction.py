import numpy as np
import rasterio
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_opening
from scipy.ndimage import label

# =========================
# INPUT / OUTPUT PATHS
# =========================

INPUT_PATH = "prediction_binary.tif"

OUTPUT_PATH = "prediction_clean.tif"

print("Loading prediction...")

with rasterio.open(INPUT_PATH) as src:

    pred = src.read(1)

    profile = src.profile

print("Prediction shape:", pred.shape)

# =========================
# STEP 1 — THRESHOLD
# =========================

print("Applying threshold...")

threshold = 0.6

pred_binary = pred > threshold

# =========================
# STEP 2 — REMOVE SMALL NOISE
# =========================

print("Removing small regions...")

pred_clean = remove_small_objects(

    pred_binary,

    min_size=200

)

# =========================
# STEP 3 — FILL GAPS
# =========================

print("Filling gaps...")

pred_clean = binary_closing(

    pred_clean,

    structure=np.ones((5, 5))

)

pred_clean = binary_opening(

    pred_clean,

    structure=np.ones((3, 3))

)

# =========================
# STEP 4 — KEEP LARGEST REGION
# =========================

print("Finding connected regions...")

labeled, num = label(pred_clean)

print("Regions detected:", num)

if num > 0:

    sizes = [

        np.sum(labeled == i)

        for i in range(1, num + 1)

    ]

    largest = np.argmax(sizes) + 1

    final = labeled == largest

else:

    final = pred_clean

# =========================
# SAVE RESULT
# =========================

profile.update(

    dtype="uint8",

    count=1

)

with rasterio.open(

    OUTPUT_PATH,

    "w",

    **profile

) as dst:

    dst.write(

        final.astype("uint8"),

        1

    )

print("Saved:", OUTPUT_PATH)