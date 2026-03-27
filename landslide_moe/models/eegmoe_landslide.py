"""
eegmoe_landslide.py
═══════════════════════════════════════════════════════════════════════════════
LandslideEEGMoE — Exact replication of EEGMoE (Gao et al., IEEE TNNLS 2026)
adapted for landslide susceptibility mapping.

Paper §III Methods, Fig. 2 (architecture) and Fig. 3 (pre-training).

Dataset folder layout:
  dataset/
    Puthumala-Training_data/
      DEM/         Copernicus_DEM_30m.tif
      Rainfall/    imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc
      Sentinel-1/  Sentinel-1_2019-12-08_SAR.tif
      Sentinel-2/  B02.tif B03.tif B04.tif B08.tif B11.tif ...
      Soil_Mositure/2019/  SM_SMAP_I_*.tif

    Wayanad_validation_data/2024-12-11/
      DEM/  Rainfall Data/  Sentinel-1/  Sentinel-2/
      Soil_moisture/Soil_Mositure/ SM_SMAP_I_*.tif

  ground_truth/  LandslideAtlas_new_2023.pdf
═══════════════════════════════════════════════════════════════════════════════
"""

import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────────
# 1.  EXPERT — Linear → GeLU → Linear   (Fig. 2 "Exp" block)
#     §III-B-1: "each expert consists of a two-layer MLP with a GELU
#     activation function in between"
# ───────────────────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(hidden_size, ffn_dim)
        self.gelu = nn.GELU()
        self.fc2  = nn.Linear(ffn_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


# ───────────────────────────────────────────────────────────────────────────────
# 2.  SPECIFIC MoE — Top-K routing   (Fig. 2 left SSMoE, Eq. 1-3)
#     g_x = W_e · x                                           (Eq.1)
#     p_i = softmax(g_x)_i                                    (Eq.2)
#     SpecMoE(x) = Σ_{i∈TopK} p_i(x) · e_i(x)                (Eq.3)
# ───────────────────────────────────────────────────────────────────────────────
class SpecificMoE(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int,
                 num_experts: int = 6, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList(
            [Expert(hidden_size, ffn_dim) for _ in range(num_experts)])
        self.router = nn.Linear(hidden_size, num_experts, bias=False)   # W_e

    def forward(self, x: torch.Tensor):
        B, S, H = x.shape
        g_x    = self.router(x)                           # (B,S,E) Eq.1
        probs  = F.softmax(g_x, dim=-1)                  # (B,S,E) Eq.2
        topk_p, topk_i = torch.topk(probs, self.top_k, dim=-1)  # (B,S,K)
        topk_p = topk_p / (topk_p.sum(-1, keepdim=True) + 1e-9)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            e_idx  = topk_i[..., k]           # (B,S)
            e_prob = topk_p[..., k:k+1]       # (B,S,1)
            for eid in range(self.num_experts):
                mask = (e_idx == eid).unsqueeze(-1).float()
                if mask.any():
                    output = output + mask * e_prob * self.experts[eid](x)
        return output, probs, topk_i           # Eq.3


# ───────────────────────────────────────────────────────────────────────────────
# 3.  SHARED MoE — Soft routing   (Fig. 2 right SSMoE, Eq. 4)
#     ShareMoE(x) = Σ_{i∈F} p_i(x) · f_i(x)                  (Eq.4)
# ───────────────────────────────────────────────────────────────────────────────
class SharedMoE(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int, num_shared: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(hidden_size, ffn_dim) for _ in range(num_shared)])
        self.router  = nn.Linear(hidden_size, num_shared, bias=False)   # W_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(self.router(x), dim=-1)       # (B,S,F)
        output = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            output = output + probs[..., i:i+1] * exp(x)
        return output                                     # Eq.4


# ───────────────────────────────────────────────────────────────────────────────
# 4.  SSMoE BLOCK   (Fig. 2 "SSMoE Block", Eq. 5)
#     SSMoE(x) = SpecMoE(x) + ShareMoE(x)                     (Eq.5)
#     Additive fusion confirmed best in Table XII ablation.
# ───────────────────────────────────────────────────────────────────────────────
class SSMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int,
                 num_specific: int = 6, top_k: int = 2, num_shared: int = 2):
        super().__init__()
        self.specific = SpecificMoE(hidden_size, ffn_dim, num_specific, top_k)
        self.shared   = SharedMoE(hidden_size, ffn_dim, num_shared)
        self.norm     = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        spec_out, rp, ti = self.specific(x)
        shar_out         = self.shared(x)
        out = self.norm(x + spec_out + shar_out)          # Eq.5
        return out, rp, ti


# ───────────────────────────────────────────────────────────────────────────────
# 5.  DOMAIN-DECOUPLED ENCODER LAYER   (Fig. 2 middle ×M)
#     Fig. 2 order (bottom→top):
#       Input tokens → Multi-Head Attention → Add&Norm → SSMoE → Add&Norm
# ───────────────────────────────────────────────────────────────────────────────
class DDEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int,
                 num_specific: int = 6, top_k: int = 2,
                 num_shared: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ssmoe = SSMoEBlock(hidden_size, ffn_dim, num_specific, top_k, num_shared)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        a, _   = self.attn(x, x, x)
        x      = self.norm1(x + self.drop(a))
        m, rp, ti = self.ssmoe(x)
        x      = self.norm2(x + self.drop(m))
        return x, rp, ti


class DomainDecoupledEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int,
                 num_layers: int = 4, num_specific: int = 6, top_k: int = 2,
                 num_shared: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DDEncoderLayer(hidden_size, num_heads, ffn_dim,
                           num_specific, top_k, num_shared, dropout)
            for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        info = []
        for layer in self.layers:
            x, rp, ti = layer(x)
            info.append((rp, ti))
        return x, info


# ───────────────────────────────────────────────────────────────────────────────
# 6.  SPATIAL & FREQUENCY ENCODER   (Fig. 2 left, §III-A)
#     Exactly as drawn: Linear1 → ReLU → Linear2 → ReLU
#     Table III: MLP hidden=64, embed=128
# ───────────────────────────────────────────────────────────────────────────────
class SpatialFrequencyEncoder(nn.Module):
    def __init__(self, in_dim: int, sf_hidden: int = 64, embed_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, sf_hidden)
        self.relu1   = nn.ReLU()
        self.linear2 = nn.Linear(sf_hidden, embed_dim)
        self.relu2   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu2(self.linear2(self.relu1(self.linear1(x))))


# ───────────────────────────────────────────────────────────────────────────────
# 7.  PATCH TOKENISER   (Fig. 2 "4D Input Data → Embeddings")
#     Splits (B,C,H,W) into non-overlapping p×p patches → token sequence
#     Analogue of EEGMoE time-window segmentation (§IV-A-2)
# ───────────────────────────────────────────────────────────────────────────────
class PatchTokeniser(nn.Module):
    def __init__(self, in_channels: int, patch_size: int,
                 sf_hidden: int = 64, embed_dim: int = 128):
        super().__init__()
        self.p  = patch_size
        in_dim  = in_channels * patch_size * patch_size
        self.sf = SpatialFrequencyEncoder(in_dim, sf_hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.p
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B,C,nH,nW,p,p)
        nH, nW = H // p, W // p
        x = x.contiguous().view(B, C, nH*nW, p, p)
        x = x.permute(0,2,1,3,4).contiguous().view(B, nH*nW, C*p*p)
        return self.sf(x)                                # (B,nP,embed_dim)


# ───────────────────────────────────────────────────────────────────────────────
# 8.  POSITIONAL ENCODING
# ───────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


# ───────────────────────────────────────────────────────────────────────────────
# 9.  LOAD-BALANCING AUXILIARY LOSS   (Eq. 7-9)
# ───────────────────────────────────────────────────────────────────────────────
def load_balancing_loss(router_probs: torch.Tensor,
                        topk_idx: torch.Tensor,
                        num_experts: int) -> torch.Tensor:
    B, S, E = router_probs.shape
    one_hot = torch.zeros(B, S, E, device=router_probs.device)
    for k in range(topk_idx.shape[-1]):
        one_hot.scatter_(-1, topk_idx[..., k:k+1], 1.0)
    h = (one_hot > 0).float().mean(dim=[0,1])   # Eq.8
    D = router_probs.mean(dim=[0,1])             # Eq.9
    return num_experts * (h * D).sum()           # Eq.7


# ───────────────────────────────────────────────────────────────────────────────
# 10.  FULL MODEL   (Fig. 2 + Fig. 3)
# ───────────────────────────────────────────────────────────────────────────────
class LandslideEEGMoE(nn.Module):
    """
    Domain-Decoupled MoE for Landslide Susceptibility Mapping.

    Input feature channels (analogous to EEGMoE frequency bands δ θ α β γ):
      ch 0 : SAR backscatter (Sentinel-1 VV or VH)
      ch 1 : NDVI  = (B08-B04)/(B08+B04)
      ch 2 : NDWI  = (B03-B08)/(B03+B08)
      ch 3 : Rainfall (cumulative / max over period)
      ch 4 : Soil Moisture (SMAP, closest to event)
     [ch 5 : DEM slope — optional, set in_channels=6]

    Hyperparameters from Table II & III (paper):
      sf_hidden=64, embed_dim=128, hidden_size=512
      num_heads=4, top_k=2, num_specific=6, num_shared=2
      mask_ratio=0.4, alpha=1e-4
    """
    def __init__(
        self,
        in_channels:  int   = 5,
        patch_size:   int   = 8,
        sf_hidden:    int   = 64,
        embed_dim:    int   = 128,
        hidden_size:  int   = 512,
        num_heads:    int   = 4,
        ffn_dim:      int   = 2048,
        num_layers:   int   = 4,
        num_specific: int   = 6,
        top_k:        int   = 2,
        num_shared:   int   = 2,
        num_classes:  int   = 2,
        mask_ratio:   float = 0.4,
        alpha:        float = 1e-4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.num_specific = num_specific
        self.top_k        = top_k
        self.mask_ratio   = mask_ratio
        self.alpha        = alpha
        self.embed_dim    = embed_dim

        # Fig. 2 left — patch tokenisation + spatial-frequency encoder
        self.tokeniser  = PatchTokeniser(in_channels, patch_size, sf_hidden, embed_dim)
        self.pos_enc    = PositionalEncoding(embed_dim, dropout=dropout)
        self.input_proj = nn.Linear(embed_dim, hidden_size)

        # Fig. 2 middle — domain-decoupled encoder ×M
        self.encoder = DomainDecoupledEncoder(
            hidden_size, num_heads, ffn_dim,
            num_layers, num_specific, top_k, num_shared, dropout)

        # Fig. 3 — reconstruction head (Stage 1)
        self.recon_head = nn.Linear(hidden_size, embed_dim)

        # §III-C-2 — linear classifier (Stage 2)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ── Stage 1 — Self-supervised pre-training   (Fig. 3, §III-C-1) ───────
    def pretrain_forward(self, x: torch.Tensor):
        B = x.shape[0]
        # Embeddings Z
        Z      = self.tokeniser(x)                  # (B,nP,embed_dim)
        Z_pos  = self.pos_enc(Z)
        Z_proj = self.input_proj(Z_pos)             # (B,nP,hidden_size)
        nP     = Z.shape[1]

        # Random masking → Masked Embeddings  (mask_ratio=0.4, Table XIII)
        num_masked = int(self.mask_ratio * nP)
        noise      = torch.rand(B, nP, device=x.device)
        shuffle    = torch.argsort(noise, dim=1)
        mask_idx   = shuffle[:, :num_masked]

        Z_masked = Z_proj.clone()
        for b in range(B):
            Z_masked[b, mask_idx[b]] = 0.0

        # Domain-decoupled encoder
        Z_hat, router_info = self.encoder(Z_masked)

        # Reconstruction  (Reconstructed Embeddings)
        Z_recon = self.recon_head(Z_hat)            # (B,nP,embed_dim)

        # L1 loss on masked tokens  (Eq. 6)
        Z_target = Z.detach()
        recon_loss = torch.tensor(0.0, device=x.device)
        for b in range(B):
            recon_loss = recon_loss + F.l1_loss(
                Z_recon[b, mask_idx[b]], Z_target[b, mask_idx[b]])
        recon_loss = recon_loss / B

        # Load-balancing loss  (Eq. 7-9)
        aux_loss = torch.tensor(0.0, device=x.device)
        for (rp, ti) in router_info:
            aux_loss = aux_loss + load_balancing_loss(rp, ti, self.num_specific)
        aux_loss = aux_loss / max(len(router_info), 1)

        # L_pretrain = L1 + α·L_aux  (Eq. 10)
        return recon_loss + self.alpha * aux_loss, recon_loss

    # ── Stage 2 — Supervised fine-tuning   (§III-C-2) ─────────────────────
    def finetune_forward(self, x: torch.Tensor) -> torch.Tensor:
        Z      = self.tokeniser(x)
        Z      = self.pos_enc(Z)
        Z_proj = self.input_proj(Z)
        out, _ = self.encoder(Z_proj)
        pooled = out.mean(dim=1)                    # global avg pool
        return self.classifier(pooled)              # (B, num_classes)

    def forward(self, x: torch.Tensor, mode: str = 'finetune'):
        if mode == 'pretrain':
            return self.pretrain_forward(x)
        return self.finetune_forward(x)


class FinetuneLoss(nn.Module):
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)


# ── sanity check ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    m = LandslideEEGMoE(
        in_channels=5, patch_size=4,
        sf_hidden=32, embed_dim=64,
        hidden_size=128, num_heads=2, ffn_dim=256,
        num_layers=2, num_specific=4, top_k=2, num_shared=2)
    x = torch.randn(2, 5, 32, 32)
    tl, rl = m(x, 'pretrain')
    lg     = m(x, 'finetune')
    print(f"params={sum(p.numel() for p in m.parameters()):,}")
    print(f"pretrain total={tl.item():.4f}  recon={rl.item():.4f}")
    print(f"finetune logits={lg.shape}  ✓")
