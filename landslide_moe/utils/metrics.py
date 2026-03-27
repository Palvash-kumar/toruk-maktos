"""
metrics.py
Evaluation metrics matching EEGMoE §IV-A-3:
  - Accuracy ± std    (primary, Tables V & VI)
  - AUROC             (Table IV — imbalanced datasets)
  - AUC-PR            (Table IV)
  - F1, Precision, Recall
"""

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              average_precision_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs  = torch.softmax(logits, dim=-1).detach().numpy()
    preds  = probs.argmax(axis=1)
    y_true = labels.numpy()

    m = {}
    m['accuracy']  = float(accuracy_score(y_true, preds))

    # AUROC / AUC-PR  (binary only for landslide detection)
    if probs.shape[1] == 2:
        pos = probs[:, 1]
        try:    m['auroc']  = float(roc_auc_score(y_true, pos))
        except: m['auroc']  = 0.5
        try:    m['auc_pr'] = float(average_precision_score(y_true, pos))
        except: m['auc_pr'] = 0.0
    else:
        try:    m['auroc']  = float(roc_auc_score(y_true, probs, multi_class='ovr', average='macro'))
        except: m['auroc']  = 0.0
        m['auc_pr'] = 0.0

    avg = 'binary' if probs.shape[1] == 2 else 'macro'
    m['f1']        = float(f1_score(y_true, preds, average=avg, zero_division=0))
    m['precision'] = float(precision_score(y_true, preds, average=avg, zero_division=0))
    m['recall']    = float(recall_score(y_true, preds, average=avg, zero_division=0))
    m['confusion_matrix'] = confusion_matrix(y_true, preds).tolist()
    return m


def print_metrics(m: dict, prefix: str = '') -> None:
    cm = m.pop('confusion_matrix', None)
    print(f"\n{prefix}{'─'*40}")
    for k, v in m.items():
        print(f"{prefix}  {k:<15}: {v:.4f}")
    if cm:
        print(f"{prefix}  Confusion Matrix:")
        for row in cm: print(f"{prefix}    {row}")
        m['confusion_matrix'] = cm
    print(f"{prefix}{'─'*40}")
