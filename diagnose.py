"""
Diagnostic — figure out WHY both models sit at the random baseline.

When a plain CNN and CoAtNet both fail to beat 1/36, the model is not the
problem; the data is. This script checks the three things that would kill
learning, in order of likelihood:

  1. Alignment   Does each clip actually contain a keystroke transient,
                 and is it in a consistent place in the window?
  2. Signal      Are same-key clips more similar to each other than to
                 different-key clips? (Is there any class signal at all?)
  3. Learnability Can a tiny net overfit a handful of clips? If not, the
                 inputs are effectively label-noise.

Outputs numbers to stdout plus two PNGs in results/:
  results/diag_spectrograms.png   montage of example clips
  results/diag_alignment.png      where the energy peak sits in each window
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn

from config import PROCESS_SR, WINDOW_PRE_MS, RESULTS_DIR, IDX_TO_KEY
from dataset import KeystrokeDataset, audio_to_melspec

RESULTS_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def load_audio(path):
    a, _ = sf.read(str(path))
    if a.ndim > 1:
        a = a[:, 0]
    return a.astype(np.float32)


# ── 1. Alignment + energy ──────────────────────────────────────────────────────

def check_alignment(ds, per_key=8):
    print('\n=== 1. ALIGNMENT / ENERGY ===')
    by_key = {}
    for path, label in ds.samples:
        by_key.setdefault(label, []).append(path)

    peak_fracs, rms_vals, flat_count = [], [], 0
    expected_frac = WINDOW_PRE_MS / 250.0   # where the press SHOULD peak
    for label, paths in by_key.items():
        for p in rng.choice(paths, size=min(per_key, len(paths)), replace=False):
            a = load_audio(p)
            if a.size == 0:
                continue
            env = np.abs(a)
            peak_fracs.append(np.argmax(env) / len(a))
            rms = float(np.sqrt(np.mean(a ** 2)))
            rms_vals.append(rms)
            # "flat" = peak barely above background → probably no transient
            if env.max() < 5 * (np.median(env) + 1e-9):
                flat_count += 1

    peak_fracs = np.array(peak_fracs)
    rms_vals   = np.array(rms_vals)
    n = len(peak_fracs)
    print(f'Sampled {n} clips')
    print(f'  RMS energy        mean {rms_vals.mean():.4f}  min {rms_vals.min():.4f}  max {rms_vals.max():.4f}')
    print(f'  Peak position     mean {peak_fracs.mean():.2f}  std {peak_fracs.std():.2f}  '
          f'(keystroke SHOULD peak near {expected_frac:.2f})')
    print(f'  Clips w/ no clear transient: {flat_count}/{n} ({100*flat_count/n:.0f}%)')
    near = np.mean(np.abs(peak_fracs - expected_frac) < 0.15)
    print(f'  Peaks within ±0.15 of expected: {100*near:.0f}%  '
          f'({"GOOD alignment" if near > 0.6 else "POOR — transients scattered / missing"})')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(peak_fracs, bins=30, range=(0, 1), color='steelblue')
    ax.axvline(expected_frac, color='red', ls='--', label=f'expected {expected_frac:.2f}')
    ax.set_xlabel('Peak position within window (0=start, 1=end)')
    ax.set_ylabel('Clip count'); ax.set_title('Where the energy peak sits in each clip')
    ax.legend(); plt.tight_layout()
    out = RESULTS_DIR / 'diag_alignment.png'
    plt.savefig(out, dpi=130); plt.close()
    print(f'  -> {out}')


# ── 2. Class signal ─────────────────────────────────────────────────────────────

def check_signal(ds, keys=('q', 'p', 'a', 'm', '5', 'z'), per_key=12):
    print('\n=== 2. CLASS SIGNAL (intra- vs inter-class spectrogram distance) ===')
    want = {k for k in keys}
    specs = {}   # key_char -> list of flattened specs
    for path, label in ds.samples:
        kc = IDX_TO_KEY[label]
        if kc in want and len(specs.get(kc, [])) < per_key:
            a = load_audio(path)
            if a.size:
                specs.setdefault(kc, []).append(audio_to_melspec(a).ravel())

    present = [k for k in keys if k in specs and len(specs[k]) >= 2]
    if len(present) < 2:
        print('  Not enough clips for the requested keys.')
        return

    means = {k: np.mean(specs[k], axis=0) for k in present}
    intra = np.mean([
        np.mean([np.linalg.norm(s - means[k]) for s in specs[k]])
        for k in present
    ])
    inter_pairs = [
        np.linalg.norm(means[a] - means[b])
        for i, a in enumerate(present) for b in present[i + 1:]
    ]
    inter = float(np.mean(inter_pairs))
    print(f'  Keys compared: {present}')
    print(f'  Intra-class spread (within a key): {intra:.3f}')
    print(f'  Inter-class spread (between keys): {inter:.3f}')
    ratio = inter / (intra + 1e-9)
    print(f'  Separability ratio inter/intra : {ratio:.2f}  '
          f'({"signal present" if ratio > 1.1 else "NO usable signal — classes overlap"})')


# ── 3. Learnability ─────────────────────────────────────────────────────────────

def check_learnability(ds, n=64, steps=300):
    print('\n=== 3. LEARNABILITY (can a tiny net memorise a few clips?) ===')
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs, ys = [], []
    for i in idx:
        x, y = ds[int(i)]
        xs.append(x); ys.append(y)
    X = torch.stack(xs)
    Y = torch.tensor(ys)
    nclass = len(torch.unique(Y))
    print(f'  Overfitting {len(X)} clips spanning {nclass} classes')

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(X[0].numel(), 256), nn.ReLU(),
        nn.Linear(256, 36),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for step in range(steps):
        opt.zero_grad()
        out = net(X)
        loss = lossf(out, Y)
        loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            acc = (out.argmax(1) == Y).float().mean().item() * 100
            print(f'    step {step:3d}  loss {loss.item():.3f}  train-acc {acc:.0f}%')
    final = (net(X).argmax(1) == Y).float().mean().item() * 100
    print(f'  Final overfit accuracy: {final:.0f}%  '
          f'({"inputs ARE distinguishable" if final > 80 else "inputs NOT distinguishable → data is the problem"})')


# ── Montage ─────────────────────────────────────────────────────────────────────

def montage(ds, keys=('q', 'p', 'a', 'm', '5', 'z'), per_key=4):
    rows = [k for k in keys]
    fig, axes = plt.subplots(len(rows), per_key, figsize=(per_key * 2, len(rows) * 2))
    by_key = {}
    for path, label in ds.samples:
        by_key.setdefault(IDX_TO_KEY[label], []).append(path)
    for r, k in enumerate(rows):
        paths = by_key.get(k, [])
        for c in range(per_key):
            ax = axes[r][c] if len(rows) > 1 else axes[c]
            ax.axis('off')
            if c < len(paths):
                a = load_audio(paths[c])
                if a.size:
                    spec = audio_to_melspec(a)[0]
                    ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
            if c == 0:
                ax.set_title(f"key '{k}'", loc='left', fontsize=10)
    plt.tight_layout()
    out = RESULTS_DIR / 'diag_spectrograms.png'
    plt.savefig(out, dpi=130); plt.close()
    print(f'\nSpectrogram montage -> {out}')


if __name__ == '__main__':
    import sys
    clips_dir = sys.argv[1] if len(sys.argv) > 1 else None
    ds = KeystrokeDataset(clips_dir=clips_dir) if clips_dir else KeystrokeDataset()
    print(f'Clips dir: {clips_dir or "data/clips (default)"}')
    print(f'Loaded {len(ds)} clips, {len(set(l for _, l in ds.samples))} classes')
    check_alignment(ds)
    check_signal(ds)
    check_learnability(ds)
    montage(ds)
    print('\nDone. Open the two PNGs in results/ to eyeball the clips.')
