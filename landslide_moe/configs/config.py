"""
config.py — All hyperparameters from EEGMoE Tables II & III
"""
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ModelConfig:
    """Table III — Hyperparameters of EEGMoE Model"""
    # Spatial & Frequency Encoder (Fig. 2 left)
    in_channels:  int = 5       # SAR, NDVI, NDWI, Rainfall, SoilMoisture
    patch_size:   int = 8       # model-level patch sub-size
    sf_hidden:    int = 64      # MLP hidden size
    embed_dim:    int = 128     # MLP embed size
    # Domain-Decoupled Encoder (Fig. 2 middle)
    hidden_size:  int = 512
    num_heads:    int = 4
    ffn_dim:      int = 2048
    num_layers:   int = 4
    # SSMoE Block (Fig. 2 right)
    num_specific: int = 6       # specific expert candidates
    top_k:        int = 2       # Top-K routing
    num_shared:   int = 2       # shared experts
    # Task
    num_classes:  int = 2       # binary: landslide / non-landslide
    mask_ratio:   float = 0.4   # Table XIII best
    alpha:        float = 1e-4  # load-balance weight
    dropout:      float = 0.1

@dataclass
class TrainConfig:
    """Table II — Implementation Details"""
    optimizer:         str   = 'AdamW'
    warmup_ratio:      float = 0.05
    # Pre-training
    pretrain_lr:       float = 1e-4
    pretrain_bs:       int   = 64
    pretrain_epochs:   int   = 5
    load_balance_w:    float = 1e-4
    # Fine-tuning
    finetune_lr:       float = 1e-3
    finetune_bs:       int   = 128
    finetune_epochs:   Dict[str,int] = field(default_factory=lambda: {
        'puthumala_training':  35,   # analogous to DEAP
        'wayanad_validation':  50,   # analogous to BCIC4-2a
        'default':             35,
    })

@dataclass
class DataConfig:
    """Feature channel mapping & data settings"""
    channels: Dict[str,int] = field(default_factory=lambda: {
        'SAR':           0,   # Sentinel-1 backscatter     ← δ-band analogue
        'NDVI':          1,   # (B08-B04)/(B08+B04)        ← α-band analogue
        'NDWI':          2,   # (B03-B08)/(B03+B08)        ← α-band analogue
        'Rainfall':      3,   # IMD/GPM max daily          ← β-band analogue
        'SoilMoisture':  4,   # SMAP closest to event      ← γ-band analogue
    })
    patch_size_data: int   = 64     # spatial patch size (pixels)
    stride:          int   = 32
    normalize:       str   = 'zscore'
    target_size:     tuple = (256, 256)
    # Folder names (matching actual project tree)
    puthumala_dir:   str   = 'Puthumala-Training_data'
    wayanad_dir:     str   = 'Wayanad_validation_data'
    gt_dir:          str   = 'ground_truth'
    sar_puthumala:   str   = 'Sentinel-1/Sentinel-1_2019-12-08_SAR.tif'
    sar_wayanad_11:  str   = 'Sentinel-1/Sentinel-1_2024-12-11_SAR.tif'
    sar_wayanad_16:  str   = 'Sentinel-1/Sentinel-1_2024-12-16_SAR.tif'
    sm_dir_put:      str   = 'Soil_Mositure/2019'              # note: typo in data
    sm_dir_way:      str   = 'Soil_moisture/Soil_Mositure'

DEFAULT_MODEL = ModelConfig()
DEFAULT_TRAIN = TrainConfig()
DEFAULT_DATA  = DataConfig()

def print_config():
    print("\n" + "="*58)
    print("  LandslideEEGMoE — Config (Tables II & III)")
    print("="*58)
    m = DEFAULT_MODEL
    t = DEFAULT_TRAIN
    print(f"\n  [Model — Table III]")
    print(f"    sf_hidden / embed_dim : {m.sf_hidden} / {m.embed_dim}")
    print(f"    hidden_size           : {m.hidden_size}")
    print(f"    num_heads / num_layers: {m.num_heads} / {m.num_layers}")
    print(f"    Top-K                 : {m.top_k}")
    print(f"    specific / shared     : {m.num_specific} / {m.num_shared}")
    print(f"    mask_ratio            : {m.mask_ratio}")
    print(f"    alpha (L_aux weight)  : {m.alpha}")
    print(f"\n  [Training — Table II]")
    print(f"    Optimizer             : {t.optimizer}")
    print(f"    Warmup ratio          : {t.warmup_ratio}")
    print(f"    Pretrain LR / BS      : {t.pretrain_lr} / {t.pretrain_bs}")
    print(f"    Finetune LR / BS      : {t.finetune_lr} / {t.finetune_bs}")
    print(f"    Pretrain epochs       : {t.pretrain_epochs}")
    print("="*58 + "\n")

if __name__ == '__main__':
    print_config()
