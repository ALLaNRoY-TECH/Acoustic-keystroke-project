"""
realign.py — salvage existing clips by re-centering each one on the actual
keystroke transient.

The collector's wall-clock timestamps drifted from the audio sample clock
(stream-start offset + dropped audio blocks over a long session), so clips
were windowed in the wrong place: the keystroke landed at a random position
in each clip (peak-position std ~0.30) and the per-key signal washed out.
Both the CNN and CoAtNet were therefore stuck at the 1/36 random baseline.

This re-extracts from the ORIGINAL recording. For each labelled keystroke it
searches a neighbourhood — bounded by the midpoints to the neighbouring
keystrokes so drift can't make it grab the wrong press — finds the transient,
and writes a window with the onset at a FIXED position in every clip.

Usage:
  python realign.py data/raw/session01.wav data/raw/session01.csv
Output:
  data/clips_aligned/<session>/<key>/<key>_NNNN.wav
"""

import sys
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from isolate import resample_audio, _load_labels
from config import PROCESS_SR, KEYS, DATA_DIR

WINDOW_SAMPLES = 4000     # 250 ms at 16 kHz — matches what dataset.py expects
ONSET_FRAC     = 0.25     # where the onset sits in every output clip
SEARCH_CAP_MS  = 300      # never look further than this from the label anchor
ENV_WIN        = 48       # ~3 ms moving-average for the amplitude envelope
BACKTRACK_REL  = 0.15     # onset = walk back from peak to 15% of its height


def envelope(a):
    """Smoothed amplitude envelope."""
    return np.convolve(np.abs(a), np.ones(ENV_WIN) / ENV_WIN, mode='same')


def find_onset(env, lo, hi):
    """Loudest point in [lo, hi], then backtrack to the rising edge."""
    peak = lo + int(np.argmax(env[lo:hi]))
    thr  = BACKTRACK_REL * env[peak]
    i = peak
    while i > lo and env[i] > thr:
        i -= 1
    return i, peak


def realign(wav_path, labels_path, out_dir):
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = resample_audio(audio, sr, PROCESS_SR)
    sr = PROCESS_SR
    env = envelope(audio)

    events = _load_labels(labels_path)          # ordered [(ts, key), ...]
    ts = [e[0] for e in events]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = int(SEARCH_CAP_MS / 1000 * sr)
    pre = int(ONSET_FRAC * WINDOW_SAMPLES)
    counts, skipped, shifts = {}, 0, []

    for i, (t, key) in enumerate(events):
        if key not in KEYS:
            continue
        c  = int(t * sr)
        lo = c - cap
        hi = c + cap
        # Bound the search by the midpoints to neighbouring keystrokes so that
        # even large drift can't snap us onto the wrong (adjacent) press.
        if i > 0:
            lo = max(lo, int((ts[i - 1] + t) / 2 * sr))
        if i < len(events) - 1:
            hi = min(hi, int((t + ts[i + 1]) / 2 * sr))
        lo = max(0, lo)
        hi = min(len(audio), hi)
        if hi - lo < ENV_WIN:
            skipped += 1
            continue

        onset, peak = find_onset(env, lo, hi)
        shifts.append((onset - c) / sr * 1000.0)   # ms the label was off by

        start = onset - pre
        end   = start + WINDOW_SAMPLES
        if start < 0 or end > len(audio):
            skipped += 1
            continue

        clip = audio[start:end].copy()
        pk = np.max(np.abs(clip))
        if pk > 1e-6:
            clip /= pk

        key_dir = out_dir / key
        key_dir.mkdir(exist_ok=True)
        idx = counts.get(key, 0)
        sf.write(str(key_dir / f'{key}_{idx:04d}.wav'), clip, sr)
        counts[key] = idx + 1

    shifts = np.array(shifts)
    return {
        'clips_saved': sum(counts.values()),
        'events_total': len(events),
        'skipped': skipped,
        'label_shift_ms': {
            'mean': round(float(shifts.mean()), 1) if shifts.size else None,
            'std':  round(float(shifts.std()), 1)  if shifts.size else None,
            'p95':  round(float(np.percentile(np.abs(shifts), 95)), 1) if shifts.size else None,
        },
        'clips_per_key': counts,
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    wav_path    = Path(sys.argv[1])
    labels_path = Path(sys.argv[2])
    session     = wav_path.stem
    out_dir     = DATA_DIR / 'clips_aligned' / session

    print(f'Re-aligning {wav_path.name} using {labels_path.name} ...')
    stats = realign(wav_path, labels_path, out_dir)
    print(json.dumps(stats, indent=2))
    print(f'\nAligned clips -> {out_dir}')
    print('Now run:  python diagnose.py data/clips_aligned')
