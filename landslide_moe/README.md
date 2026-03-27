# LandslideEEGMoE
### Domain-Decoupled Mixture-of-Experts for Landslide Susceptibility Mapping

Exact implementation of **EEGMoE** (Gao et al., IEEE TNNLS 2026) — Fig. 2 & 3 —
adapted for your real dataset structure.

---

## Architecture (Fig. 2 + Fig. 3)

```
                        ┌─────────────────────────────────────────────────────┐
(B, 5, H, W)            │            LandslideEEGMoE                         │
5-channel raster        │                                                     │
  ch0: SAR          ──► │  PatchTokeniser  ──►  PosEnc  ──►  InputProj       │
  ch1: NDVI             │  [Linear1→ReLU→Linear2→ReLU]  (Fig.2 left block)   │
  ch2: NDWI             │                 ↓ Embeddings Z                     │
  ch3: Rainfall         │    ┌────────────────────────────────────────────┐   │
  ch4: Soil Moisture    │    │   Domain-Decoupled Encoder  ×M  (Fig.2)   │   │
                        │    │  ┌────────────────────────────────────┐   │   │
                        │    │  │  Multi-Head Self-Attention          │   │   │
                        │    │  │       + Add & Norm                  │   │   │
                        │    │  ├────────────────────────────────────┤   │   │
                        │    │  │         SSMoE Block                 │   │   │
                        │    │  │  SpecificMoE (Top-K=2, 6 experts)  │   │   │
                        │    │  │        +  (Eq. 5)                  │   │   │
                        │    │  │  SharedMoE  (soft, 2 experts)      │   │   │
                        │    │  │       + Add & Norm                  │   │   │
                        │    │  └────────────────────────────────────┘   │   │
                        │    └────────────────────────────────────────────┘   │
                        │                 ↓                                   │
                        │  Stage 1: ReconHead → L1 + α·L_aux  (Fig. 3)      │
                        │  Stage 2: LinearClassifier → CrossEntropy          │
                        └─────────────────────────────────────────────────────┘
```

---

## Your Dataset → Feature Channels

| EEGMoE band | Your file | Channel |
|---|---|---|
| δ (delta) | `Sentinel-1/Sentinel-1_*_SAR.tif` band 1 | ch 0: SAR |
| θ (theta) | `Sentinel-2/B08.tif`, `B04.tif` | ch 1: NDVI |
| α (alpha) | `Sentinel-2/B03.tif`, `B08.tif` | ch 2: NDWI |
| β (beta)  | `Rainfall/*.nc` or `Rainfall Data/*.nc` | ch 3: Rainfall |
| γ (gamma) | `Soil_Mositure/*/SM_SMAP_I_*.tif` | ch 4: Soil Moisture |

---

## File Structure

```
toruk-maktos/              ← your project root
├── dataset/
│   ├── Puthumala-Training_data/
│   │   ├── DEM/Copernicus_DEM_30m.tif
│   │   ├── Rainfall/imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc
│   │   ├── Sentinel-1/Sentinel-1_2019-12-08_SAR.tif
│   │   ├── Sentinel-2/B02.tif B03.tif B04.tif B08.tif ...
│   │   └── Soil_Mositure/2019/SM_SMAP_I_*.tif
│   └── Wayanad_validation_data/
│       ├── 2024-12-11/
│       │   ├── DEM/  Rainfall Data/  Sentinel-1/  Sentinel-2/
│       │   └── Soil_moisture/Soil_Mositure/SM_SMAP_I_*.tif
│       └── 2024-12-16/
│           ├── DEM/  Sentinel-1/  Sentinel-2/
│
├── ground_truth/
│   └── LandslideAtlas_new_2023.pdf  ← extract binary raster from this
│
└── landslide_moe/                   ← THIS CODEBASE
    ├── models/eegmoe_landslide.py   ← Full architecture (§III)
    ├── data/data_pipeline.py        ← Real data loader (matching your tree)
    ├── train/train.py               ← Two-stage training (Table II)
    ├── utils/metrics.py             ← Acc, AUROC, AUC-PR (§IV-A-3)
    └── configs/config.py            ← Tables II & III hyperparameters
```

---

## Equations Implemented

| Eq. | Description | Location |
|---|---|---|
| 1–2 | Router score & softmax | `SpecificMoE.forward()` |
| 3 | SpecMoE Top-K output | `SpecificMoE.forward()` |
| 4 | ShareMoE soft output | `SharedMoE.forward()` |
| 5 | SSMoE = SpecMoE + ShareMoE (additive) | `SSMoEBlock.forward()` |
| 6 | L1 reconstruction loss | `pretrain_forward()` |
| 7–9 | Load-balancing auxiliary loss | `load_balancing_loss()` |
| 10 | L_pretrain = L1 + α·L_aux | `pretrain_forward()` |
| 11 | L_cls = cross-entropy | `FinetuneLoss` |

---

## Hyperparameters (Tables II & III — verbatim)

| Parameter | Value |
|---|---|
| MLP hidden / embed | 64 / 128 |
| Hidden size | 512 |
| Attention heads | 4 |
| Top-K | 2 |
| Specific experts | 6 |
| Shared experts | 2 |
| Mask ratio | 0.4 |
| α (load-balance) | 1e-4 |
| Optimizer | AdamW |
| Warmup ratio | 0.05 |
| Pretrain LR / BS | 1e-4 / 64 |
| Finetune LR / BS | 1e-3 / 128 |
| Pretrain epochs | 5 |
| Finetune epochs | 35 (Puthumala) / 50 (Wayanad) |

---

## Quick Start

```python
# Install dependencies
pip install torch rasterio netCDF4 scipy scikit-learn

# Architecture test
python landslide_moe/models/eegmoe_landslide.py

# Train with synthetic data (no real files needed)
python landslide_moe/train/train.py

# Train with REAL data
python landslide_moe/train/train.py --use_real_data \
    --dataset_root dataset \
    --pretrain_epochs 5 \
    --finetune_epochs 35
```

---

## Notes on Real Data
- **Ground truth**: Extract binary raster from `LandslideAtlas_new_2023.pdf`
  using a GIS tool (QGIS / ArcGIS) — save as `label_puthumala.tif` and `label_wayanad.tif`
- **Rainfall folder** for Wayanad has a space: `"Rainfall Data/"` — handled automatically
- **Soil moisture typo**: folder is `Soil_Mositure` (not `Moisture`) — handled automatically
- **Multiple SM dates**: code picks the `.tif` file closest to the event date automatically
