"""
train.py
═══════════════════════════════════════════════════════════════════════════════
Two-stage training for LandslideEEGMoE.
Mirrors EEGMoE §III-C exactly.

Table II hyperparameters used verbatim:
  Optimizer        : AdamW
  Warmup ratio     : 0.05
  Base LR pretrain : 1e-4
  Base LR finetune : 1e-3
  Batch pretrain   : 64
  Batch finetune   : 128
  Epochs pretrain  : 5
  Load balance α   : 1e-4  (set in model, not here)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, math, argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from models.eegmoe_landslide import LandslideEEGMoE, FinetuneLoss
from data.data_pipeline import (get_pretrain_loader, get_finetune_loaders,
                                  generate_synthetic_data, prepare_datasets)
from data.generate_labels import load_or_generate_labels
from utils.metrics import compute_metrics, print_metrics


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE — linear warmup + cosine decay  (Table II: warmup_ratio=0.05)
# ══════════════════════════════════════════════════════════════════════════════
def cosine_with_warmup(optimizer, warmup_steps: int,
                       total_steps: int) -> LambdaLR:
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))
    return LambdaLR(optimizer, lr_fn)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Self-Supervised Pre-training  (§III-C-1, Fig. 3)
# L_pretrain = L1 + α·L_aux  (Eq. 10)
# ══════════════════════════════════════════════════════════════════════════════
def pretrain(model: LandslideEEGMoE,
             loader: DataLoader,
             device: torch.device,
             num_epochs:   int   = 5,      # Table II
             base_lr:      float = 1e-4,   # Table II
             warmup_ratio: float = 0.05,   # Table II
             save_path:    str   = 'pretrained.pt') -> None:

    model.to(device).train()
    opt   = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    total = len(loader) * num_epochs
    sched = cosine_with_warmup(opt, int(total * warmup_ratio), total)

    print(f"\n{'═'*58}")
    print(f"  STAGE 1 — Self-Supervised Pre-training  (Fig. 3)")
    print(f"  Epochs={num_epochs}  LR={base_lr}  Batches/epoch={len(loader)}")
    print(f"{'═'*58}")

    for epoch in range(1, num_epochs + 1):
        tot_loss = rec_loss = n = 0
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list,tuple)) else batch).to(device)
            opt.zero_grad()
            tl, rl = model(x, mode='pretrain')
            tl.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot_loss += tl.item(); rec_loss += rl.item(); n += 1

        print(f"  Epoch [{epoch:02d}/{num_epochs}]  "
              f"L_total={tot_loss/n:.4f}  L1={rec_loss/n:.4f}  "
              f"LR={sched.get_last_lr()[0]:.2e}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n  ✓ Pre-trained weights → {save_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Supervised Fine-tuning  (§III-C-2)
# L_cls = cross-entropy  (Eq. 11)
# ══════════════════════════════════════════════════════════════════════════════
def finetune(model: LandslideEEGMoE,
             train_loader:  DataLoader,
             val_loader:    DataLoader,
             device:        torch.device,
             num_epochs:    int   = 35,    # Table II (DEAP equiv.)
             base_lr:       float = 1e-3,  # Table II
             warmup_ratio:  float = 0.05,
             pretrain_path: Optional[str] = None,
             save_path:     str   = 'finetuned.pt') -> dict:

    # Inherit pre-trained weights  (§III-C-2)
    if pretrain_path and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded pre-trained weights from {pretrain_path}")
        if missing:     print(f"  Missing  : {missing[:3]} ...")
        if unexpected:  print(f"  Unexpected: {unexpected[:3]} ...")
    else:
        print("  [INFO] No pretrained weights found — training from scratch.")

    model.to(device)
    criterion = FinetuneLoss()
    opt   = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    total = len(train_loader) * num_epochs
    sched = cosine_with_warmup(opt, int(total * warmup_ratio), total)

    print(f"\n{'═'*58}")
    print(f"  STAGE 2 — Supervised Fine-tuning  (§III-C-2)")
    print(f"  Epochs={num_epochs}  LR={base_lr}")
    print(f"{'═'*58}")

    history    = {'train_loss': [], 'val_acc': [], 'val_auroc': []}
    best_acc   = 0.0

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tloss = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x, mode='finetune'), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tloss += loss.item(); n += 1

        avg = tloss / max(n, 1)

        # ── Validate ───────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, device)
        history['train_loss'].append(avg)
        history['val_acc'].append(metrics['accuracy'])
        history['val_auroc'].append(metrics.get('auroc', 0.0))

        print(f"  Epoch [{epoch:02d}/{num_epochs}]  "
              f"TrainLoss={avg:.4f}  "
              f"ValAcc={metrics['accuracy']:.4f}  "
              f"ValAUROC={metrics.get('auroc',0):.4f}")

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"\n  ✓ Best Val Acc={best_acc:.4f}  → {save_path}\n")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model: LandslideEEGMoE,
             loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device), mode='finetune').cpu())
            labels_all.append(y.cpu())
    return compute_metrics(
        torch.cat(logits_all), torch.cat(labels_all))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def build_model(args) -> LandslideEEGMoE:
    return LandslideEEGMoE(
        in_channels  = args.in_channels,
        patch_size   = args.patch_size,
        sf_hidden    = args.sf_hidden,
        embed_dim    = args.embed_dim,
        hidden_size  = args.hidden_size,
        num_heads    = args.num_heads,
        ffn_dim      = args.ffn_dim,
        num_layers   = args.num_layers,
        num_specific = args.num_specific,
        top_k        = args.top_k,
        num_shared   = args.num_shared,
        num_classes  = args.num_classes,
        mask_ratio   = args.mask_ratio,
        alpha        = args.alpha,
        dropout      = args.dropout,
    )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = build_model(args)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load data ──────────────────────────────────────────────────────────
    if args.use_real_data and os.path.isdir(args.dataset_root):
        H, W = args.target_h, args.target_w

        # Load ground-truth .tif if available, else generate heuristic labels
        print("\nLoading / generating Puthumala labels...")
        label_put = load_or_generate_labels(
            args.dataset_root, 'puthumala',
            target_size=(H, W),
            label_tif_path=args.label_puthumala,
            threshold=args.label_threshold,
        )
        print("Loading / generating Wayanad labels...")
        label_way = load_or_generate_labels(
            args.dataset_root, 'wayanad',
            target_size=(H, W),
            label_tif_path=args.label_wayanad,
            threshold=args.label_threshold,
        )

        patches, labels = prepare_datasets(
            args.dataset_root, label_put, label_way,
            patch_size  = args.patch_size_data,
            stride      = args.stride,
            target_size = (H, W),
        )
    else:
        print("[INFO] Using synthetic data (set --use_real_data for real data)")
        patches, labels = generate_synthetic_data(
            n=args.num_samples, C=args.in_channels, p=args.patch_size_data)

    # ── Stage 1 ────────────────────────────────────────────────────────────
    if args.do_pretrain:
        loader = get_pretrain_loader(patches, args.pretrain_bs)
        pretrain(model, loader, device,
                 num_epochs=args.pretrain_epochs,
                 base_lr=args.pretrain_lr,
                 save_path=args.pretrain_save)

    # ── Stage 2 ────────────────────────────────────────────────────────────
    if args.do_finetune:
        train_l, val_l, test_l = get_finetune_loaders(
            patches, labels, batch_size=args.finetune_bs)
        finetune(model, train_l, val_l, device,
                 num_epochs=args.finetune_epochs,
                 base_lr=args.finetune_lr,
                 pretrain_path=args.pretrain_save,
                 save_path=args.finetune_save)

        # Final test
        state = torch.load(args.finetune_save, map_location=device)
        model.load_state_dict(state)
        m = evaluate(model, test_l, device)
        print("\n── Final Test Results ──────────────────────────────")
        print_metrics(m, prefix='  ')


# ── CLI defaults (all matching Table II & III exactly) ─────────────────────────
def get_parser():
    p = argparse.ArgumentParser('LandslideEEGMoE')
    # Model — Table III
    p.add_argument('--in_channels',  type=int,   default=5)
    p.add_argument('--patch_size',   type=int,   default=8)   # model patch
    p.add_argument('--sf_hidden',    type=int,   default=64)
    p.add_argument('--embed_dim',    type=int,   default=128)
    p.add_argument('--hidden_size',  type=int,   default=512)
    p.add_argument('--num_heads',    type=int,   default=4)
    p.add_argument('--ffn_dim',      type=int,   default=2048)
    p.add_argument('--num_layers',   type=int,   default=4)
    p.add_argument('--num_specific', type=int,   default=6)
    p.add_argument('--top_k',        type=int,   default=2)
    p.add_argument('--num_shared',   type=int,   default=2)
    p.add_argument('--num_classes',  type=int,   default=2)
    p.add_argument('--mask_ratio',   type=float, default=0.4)
    p.add_argument('--alpha',        type=float, default=1e-4)
    p.add_argument('--dropout',      type=float, default=0.1)
    # Training — Table II
    p.add_argument('--pretrain_epochs', type=int,   default=5)
    p.add_argument('--finetune_epochs', type=int,   default=35)
    p.add_argument('--pretrain_lr',     type=float, default=1e-4)
    p.add_argument('--finetune_lr',     type=float, default=1e-3)
    p.add_argument('--pretrain_bs',     type=int,   default=64)
    p.add_argument('--finetune_bs',     type=int,   default=128)
    p.add_argument('--warmup_ratio',    type=float, default=0.05)
    p.add_argument('--do_pretrain',  action='store_true', default=True)
    p.add_argument('--do_finetune',  action='store_true', default=True)
    # Data
    p.add_argument('--use_real_data',   action='store_true', default=False)
    p.add_argument('--dataset_root',    type=str,   default='dataset')
    p.add_argument('--patch_size_data', type=int,   default=64)
    p.add_argument('--stride',          type=int,   default=32)
    p.add_argument('--target_h',        type=int,   default=256)
    p.add_argument('--target_w',        type=int,   default=256)
    p.add_argument('--num_samples',     type=int,   default=500)
    # Labels
    p.add_argument('--label_puthumala', type=str, default='',
                   help='Path to ground-truth label .tif for Puthumala (optional)')
    p.add_argument('--label_wayanad',  type=str, default='',
                   help='Path to ground-truth label .tif for Wayanad (optional)')
    p.add_argument('--label_threshold', type=float, default=0.55,
                   help='Heuristic label threshold (only used if no .tif provided)')
    # Paths
    p.add_argument('--pretrain_save', type=str, default='checkpoints/pretrained.pt')
    p.add_argument('--finetune_save', type=str, default='checkpoints/finetuned.pt')
    return p


if __name__ == '__main__':
    args = get_parser().parse_args([])   # defaults
    main(args)
