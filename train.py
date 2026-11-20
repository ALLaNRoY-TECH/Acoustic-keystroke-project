"""
Stage D — Training.

Usage:
  python train.py --model baseline          # fast first run
  python train.py --model coatnet           # reproduce the paper result
  python train.py --model coatnet --epochs 100 --lr 1e-4
  python train.py --model baseline --smoke  # 2-epoch smoke-test

Outputs:
  saved_models/best_<model>.pt
  results/history_<model>.json
  results/preds_<model>.npy   (test-set predictions)
  results/labels_<model>.npy  (test-set true labels)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    VAL_SPLIT, TEST_SPLIT, SEED,
    FREQ_MASK_PARAM, TIME_MASK_PARAM, NUM_MASKS,
    AUG_MAX_SHIFT, AUG_NOISE_STD, AUG_CONTRAST, AUG_PROB,
    NUM_CLASSES, MODELS_DIR, RESULTS_DIR,
)
from dataset import KeystrokeDataset
from models import build_model


# ── SpecAugment ───────────────────────────────────────────────────────────────

class SpecAugment:
    """
    Train-time augmentation for 1xHxW log-mel spectrograms.

    SpecAugment masking (Park et al. 2019) PLUS three effects that simulate the
    cross-session variation measured in this project, so the model generalises
    to new recordings instead of memorising one session's fingerprint:
      * time-roll  — onset-position jitter
      * contrast   — gain/offset (per-clip normalisation differences)
      * noise      — noise-floor differences
    """

    def __init__(self, freq_mask=FREQ_MASK_PARAM, time_mask=TIME_MASK_PARAM,
                 num_masks=NUM_MASKS, max_shift=AUG_MAX_SHIFT,
                 noise_std=AUG_NOISE_STD, contrast=AUG_CONTRAST, prob=AUG_PROB):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.contrast  = contrast
        self.prob      = prob

    def __call__(self, x):
        x = x.clone()
        _, H, W = x.shape

        # Time-roll — onset-position jitter across sessions.
        if self.max_shift and np.random.rand() < self.prob:
            s = int(np.random.randint(-self.max_shift, self.max_shift + 1))
            x = torch.roll(x, shifts=s, dims=2)
            if s > 0:
                x[:, :, :s] = 0.0
            elif s < 0:
                x[:, :, s:] = 0.0

        # Contrast / brightness — per-clip normalisation differences.
        if self.contrast and np.random.rand() < self.prob:
            g = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            b = np.random.uniform(-self.contrast, self.contrast)
            x = (x * g + b).clamp(0.0, 1.0)

        # Additive noise — noise-floor differences.
        if self.noise_std and np.random.rand() < self.prob:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)

        # SpecAugment frequency/time masks.
        for _ in range(self.num_masks):
            f  = np.random.randint(0, self.freq_mask + 1)
            f0 = np.random.randint(0, max(1, H - f + 1))
            x[:, f0: f0 + f, :] = 0.0

            t  = np.random.randint(0, self.time_mask + 1)
            t0 = np.random.randint(0, max(1, W - t + 1))
            x[:, :, t0: t0 + t] = 0.0
        return x


# ── Dataset splits ────────────────────────────────────────────────────────────

class _TransformWrapper(Dataset):
    """Applies an optional transform to a Subset."""

    def __init__(self, subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        if self.transform:
            x = self.transform(x)
        return x, y


def make_splits(dataset, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    labels  = [dataset.samples[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    sss1 = StratifiedShuffleSplit(1, test_size=test_split, random_state=seed)
    tv_idx, test_idx = next(sss1.split(indices, labels))

    tv_labels = [labels[i] for i in tv_idx]
    val_frac  = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(1, test_size=val_frac, random_state=seed)
    tr_sub, val_sub = next(sss2.split(tv_idx, tv_labels))

    train_idx = [tv_idx[i] for i in tr_sub]
    val_idx   = [tv_idx[i] for i in val_sub]

    from torch.utils.data import Subset
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


# ── Accuracy ──────────────────────────────────────────────────────────────────

def topk_accuracy(outputs, labels, topk=(1, 5)):
    with torch.no_grad():
        maxk      = max(topk)
        _, pred   = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        correct   = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
        results   = []
        for k in topk:
            ck = correct[:k].reshape(-1).float().sum()
            results.append(ck.mul_(100.0 / labels.size(0)).item())
    return results


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model_type='baseline', epochs=EPOCHS, lr=None, smoke=False, clips_dir=None):
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    dataset = KeystrokeDataset(clips_dir=clips_dir) if clips_dir else KeystrokeDataset()
    print(f'Dataset: {len(dataset)} clips')
    print('Class counts:', dataset.class_counts())

    train_ds, val_ds, test_ds = make_splits(dataset)
    augment = SpecAugment()

    train_loader = DataLoader(
        _TransformWrapper(train_ds, augment),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}')

    model     = build_model(model_type, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    # Pretrained CoAtNet fine-tunes at a much lower LR than the from-scratch CNN.
    if lr is None:
        lr = 1e-4 if model_type == 'coatnet' else LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    print(f'Optimizer: AdamW  lr={lr:g}  weight_decay={WEIGHT_DECAY:g}')

    n_epochs  = 2 if smoke else epochs
    # CoAtNet needs an LR warmup: the randomly-initialised head otherwise emits
    # large early gradients that corrupt the pretrained backbone, stalling the
    # model at the random baseline (exactly what lr=1e-3 with no warmup did).
    warmup_epochs = 5 if (model_type == 'coatnet' and not smoke) else 0
    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.02, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history      = {'train_loss': [], 'val_acc1': [], 'val_acc5': []}
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── validate ──
        model.eval()
        acc1s, acc5s = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                a1, a5 = topk_accuracy(model(x), y)
                acc1s.append(a1); acc5s.append(a5)

        avg_loss = running_loss / len(train_loader)
        val_acc1 = float(np.mean(acc1s))
        val_acc5 = float(np.mean(acc5s))
        history['train_loss'].append(avg_loss)
        history['val_acc1'].append(val_acc1)
        history['val_acc5'].append(val_acc5)

        print(f'Epoch {epoch:3d}/{n_epochs} | loss {avg_loss:.4f} | '
              f'val top-1 {val_acc1:.1f}% | val top-5 {val_acc5:.1f}%')

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save({
                'epoch': epoch, 'model_type': model_type,
                'model_state': model.state_dict(),
                'val_acc1': val_acc1, 'val_acc5': val_acc5,
            }, MODELS_DIR / f'best_{model_type}.pt')

        scheduler.step()

    # ── test ──
    ckpt = torch.load(MODELS_DIR / f'best_{model_type}.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    all_preds, all_labels = [], []
    top5_correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            top5   = logits.topk(5, dim=1).indices
            top5_correct += top5.eq(y.view(-1, 1)).any(dim=1).sum().item()
            total        += y.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Exact per-sample accuracy (not an unweighted mean of per-batch numbers).
    test_acc1 = 100.0 * float((all_preds == all_labels).mean())
    test_acc5 = 100.0 * top5_correct / total
    print(f'\nTest top-1: {test_acc1:.1f}%  |  Test top-5: {test_acc5:.1f}%')

    # ── save results ──
    with open(RESULTS_DIR / f'history_{model_type}.json', 'w') as f:
        json.dump(history, f, indent=2)

    results = {
        'model_type': model_type,
        'test_top1': test_acc1, 'test_top5': test_acc5,
        'best_val_top1': best_val_acc,
    }
    with open(RESULTS_DIR / f'results_{model_type}.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.save(RESULTS_DIR / f'preds_{model_type}.npy',  np.array(all_preds))
    np.save(RESULTS_DIR / f'labels_{model_type}.npy', np.array(all_labels))

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',  default='baseline', choices=['baseline', 'coatnet'])
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--lr',     type=float, default=None,
                   help='learning rate (default: 1e-3 baseline, 1e-4 coatnet)')
    p.add_argument('--smoke',  action='store_true', help='2-epoch test run')
    p.add_argument('--clips',  default=None, help='clips dir (default data/clips)')
    args = p.parse_args()
    train(args.model, args.epochs, args.lr, args.smoke, args.clips)
