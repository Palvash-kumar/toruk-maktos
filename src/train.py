import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np

from preprocessing import (
    load_all_data,
    create_features,
    create_labels
)

from dataset import LandslideDataset
from model.model import EEGMoEModel
import config


# -----------------------------
# DEVICE SETUP
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# -----------------------------
# PATH CHECK
# -----------------------------

print("Current working directory:")
print(os.getcwd())

print(
    "File exists:",
    os.path.exists(
        "data1/DEM/Copernicus.tif"
    )
)


# -----------------------------
# LOAD DATA
# -----------------------------

print("\nLoading data...")

dem, s1, s2, rainfall, soil = load_all_data()


# -----------------------------
# CREATE FEATURES
# -----------------------------

print("Creating features...")

features = create_features(
    s1,
    s2,
    rainfall,
    soil
)


# -----------------------------
# CREATE LABELS
# -----------------------------

print("Creating labels...")

labels = create_labels(
    rainfall,
    soil,
    dem
)


# -----------------------------
# SAFETY CHECK FOR NaN
# -----------------------------

def check_nan(name, arr):

    print("\nChecking:", name)

    print(
        "NaN count:",
        np.isnan(arr).sum()
    )

    print(
        "Min:",
        np.min(arr)
    )

    print(
        "Max:",
        np.max(arr)
    )


check_nan("Features", features)
check_nan("Labels", labels)


# -----------------------------
# DATASET
# -----------------------------

dataset = LandslideDataset(
    features,
    labels
)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# MODEL
# -----------------------------

model = EEGMoEModel(
    config.INPUT_DIM
)

model.to(device)


# -----------------------------
# LOSS
# -----------------------------

loss_fn = nn.BCEWithLogitsLoss()


# -----------------------------
# OPTIMIZER
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)


print("\nStarting training\n")


# -----------------------------
# TRAINING LOOP
# -----------------------------

for epoch in range(config.EPOCHS):

    model.train()

    total_loss = 0

    for X, y in loader:

        # Move to device
        X = X.to(device)
        y = y.to(device)

        # Ensure float dtype
        y = y.float()

        optimizer.zero_grad()

        # Forward
        pred = model(X)

        # Loss
        loss = loss_fn(
            pred.squeeze(),
            y
        )

        # Check NaN loss
        if torch.isnan(loss):

            print("NaN loss detected!")
            print(
                "Prediction min:",
                pred.min().item()
            )
            print(
                "Prediction max:",
                pred.max().item()
            )

            break

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{config.EPOCHS} | "
        f"Loss: {avg_loss:.4f}"
    )


# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nTraining complete")
print("Model saved as model.pth")