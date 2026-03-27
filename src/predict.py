import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import (
    load_all_data,
    create_features
)

from model.model import EEGMoEModel
import config


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# -----------------------------
# LOAD DATA
# -----------------------------

print("Loading data...")

dem, s1, s2, rainfall, soil = load_all_data()


# -----------------------------
# CREATE FEATURES
# -----------------------------

features = create_features(
    s1,
    s2,
    rainfall,
    soil
)


# -----------------------------
# LOAD MODEL
# -----------------------------

model = EEGMoEModel(
    config.INPUT_DIM
)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded")


# -----------------------------
# PREDICT
# -----------------------------

X = torch.tensor(
    features,
    dtype=torch.float32
)

X = X.view(
    -1,
    config.INPUT_DIM
)

X = X.to(device)

with torch.no_grad():

    pred = model(X)

    prob = torch.sigmoid(pred)

    prob = prob.cpu().numpy()


# -----------------------------
# RESHAPE TO MAP
# -----------------------------

height, width, _ = features.shape

risk_map = prob.reshape(
    height,
    width
)

print("Prediction complete")

# -----------------------------
# SHOW MAP
# -----------------------------

plt.imshow(
    risk_map,
    cmap="hot"
)

plt.colorbar()

plt.title(
    "Landslide Risk Map"
)

plt.show()