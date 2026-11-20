"""
Stage F — Evaluation.

Generates confusion matrix, per-key accuracy, and training curves.
Run after train.py has completed.

Usage:
  python evaluate.py --model baseline
  python evaluate.py --model coatnet
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-safe; change to 'TkAgg' if you want interactive windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import KEYS, NUM_CLASSES, RESULTS_DIR


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, model_type):
    cm      = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=False, cmap='Blues',
        xticklabels=KEYS, yticklabels=KEYS,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Normalised Confusion Matrix — {model_type}')
    plt.tight_layout()

    out = RESULTS_DIR / f'confusion_{model_type}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Confusion matrix saved -> {out}')


def top_confused_pairs(preds, labels, n=10):
    cm    = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], KEYS[i], KEYS[j]))
    pairs.sort(reverse=True)
    print(f'\nTop {n} confused pairs (true -> predicted, count):')
    for count, true_k, pred_k in pairs[:n]:
        print(f'  {true_k!r} -> {pred_k!r}  ({count}x)')


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(model_type):
    hist_path = RESULTS_DIR / f'history_{model_type}.json'
    if not hist_path.exists():
        print(f'No history file found at {hist_path}')
        return

    with open(hist_path) as f:
        h = json.load(f)

    epochs = list(range(1, len(h['train_loss']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, h['train_loss'], label='train loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training loss')
    axes[0].legend()

    axes[1].plot(epochs, h['val_acc1'], label='val top-1')
    axes[1].plot(epochs, h['val_acc5'], label='val top-5')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation accuracy')
    axes[1].legend()

    plt.tight_layout()
    out = RESULTS_DIR / f'training_curves_{model_type}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Training curves saved -> {out}')


# ── Per-key accuracy bar chart ────────────────────────────────────────────────

def plot_per_key_accuracy(preds, labels, model_type):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    per_key_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(KEYS, per_key_acc, color='steelblue')
    ax.axhline(per_key_acc.mean(), color='red', linestyle='--', label=f'mean {per_key_acc.mean():.1f}%')
    ax.set_xlabel('Key'); ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_title(f'Per-key accuracy — {model_type}')
    ax.legend()
    plt.tight_layout()

    out = RESULTS_DIR / f'per_key_accuracy_{model_type}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Per-key accuracy chart saved -> {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(model_type):
    RESULTS_DIR.mkdir(exist_ok=True)

    preds_path  = RESULTS_DIR / f'preds_{model_type}.npy'
    labels_path = RESULTS_DIR / f'labels_{model_type}.npy'

    if not preds_path.exists():
        print(f'No predictions found. Run: python train.py --model {model_type}')
        return

    preds  = np.load(preds_path)
    labels = np.load(labels_path)

    # Summary
    top1 = (preds == labels).mean() * 100
    print(f'Model: {model_type}')
    print(f'Test top-1 accuracy: {top1:.2f}%')

    # top-5 requires full logits; load from the JSON saved by train.py
    res_path = RESULTS_DIR / f'results_{model_type}.json'
    if res_path.exists():
        with open(res_path) as f:
            saved = json.load(f)
        print(f'Test top-5 accuracy: {saved.get("test_top5", "N/A")}%')
    print()
    print(classification_report(labels, preds, target_names=KEYS, digits=3))

    top_confused_pairs(preds, labels)

    # Plots
    plot_confusion_matrix(preds, labels, model_type)
    plot_per_key_accuracy(preds, labels, model_type)
    plot_training_curves(model_type)



if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='baseline', choices=['baseline', 'coatnet'])
    args = p.parse_args()
    evaluate(args.model)
