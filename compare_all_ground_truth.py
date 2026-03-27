import os
import numpy as np
import rasterio
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from skimage.transform import resize

# =========================
# PATHS
# =========================

PRED_PATH = "prediction_clean.tif"

GT_FOLDER = "ground_truth_masks"

# =========================
# LOAD PREDICTION
# =========================

print("Loading prediction")

with rasterio.open(PRED_PATH) as src:

    pred = src.read(1)

print("Prediction shape:", pred.shape)

pred = (pred > 0).astype(np.uint8)

# =========================
# METRIC STORAGE
# =========================

accuracies = []
precisions = []
recalls = []
f1_scores = []
ious = []

# =========================
# LOOP THROUGH GROUND TRUTH
# =========================

files = os.listdir(GT_FOLDER)

for file in files:

    if not file.endswith(".tif"):
        continue

    print("\nProcessing:", file)

    path = os.path.join(
        GT_FOLDER,
        file
    )

    with rasterio.open(path) as src:

        gt = src.read(1)

    print("GT shape:", gt.shape)

    # =====================
    # RESIZE IF NEEDED
    # =====================

    if gt.shape != pred.shape:

        print("Resizing ground truth")

        gt = resize(
            gt,
            pred.shape,
            preserve_range=True
        )

        gt = (
            gt > 0.5
        ).astype(np.uint8)

    else:

        gt = (
            gt > 0
        ).astype(np.uint8)

    # =====================
    # FLATTEN
    # =====================

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # =====================
    # METRICS
    # =====================

    accuracy = accuracy_score(
        gt_flat,
        pred_flat
    )

    precision = precision_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )

    recall = recall_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )

    f1 = f1_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    iou = intersection / union if union != 0 else 0

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("IoU      :", iou)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    ious.append(iou)

# =========================
# AVERAGE RESULTS
# =========================

print("\n==========================")
print("AVERAGE RESULTS")
print("==========================")

print("Accuracy :", np.mean(accuracies))
print("Precision:", np.mean(precisions))
print("Recall   :", np.mean(recalls))
print("F1 Score :", np.mean(f1_scores))
print("IoU      :", np.mean(ious))