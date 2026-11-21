"""
Stage B — Keystroke isolation.

Extracts per-keystroke WAV clips from a continuous recording.

With ground-truth labels (from collect_data.py):
  python isolate.py data/raw/session01.wav data/raw/session01.csv

Onset-detection only (no labels — e.g., a Zoom recording):
  python isolate.py data/raw/zoom_session.wav

Clips are saved to data/clips/<session_name>/<key>/<key>_NNNN.wav
"""

import csv
import sys
import json
from math import gcd
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly

from config import SAMPLE_RATE, PROCESS_SR, WINDOW_PRE_MS, WINDOW_POST_MS, CLIPS_DIR, KEYS


# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


# ── Ground-truth extraction ────────────────────────────────────────────────────

def extract_clips(wav_path, labels_path, output_dir):
    """
    Extract keystroke clips using synchronized ground-truth timestamps.
    Returns a stats dict with per-key clip counts and skip information.
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = resample_audio(audio, sr, PROCESS_SR)
    sr = PROCESS_SR

    pre  = int(WINDOW_PRE_MS  / 1000 * sr)
    post = int(WINDOW_POST_MS / 1000 * sr)

    events = _load_labels(labels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts  = {}
    skipped = 0

    for ts, key in events:
        if key not in KEYS:
            continue

        center = int(ts * sr)
        start  = center - pre
        end    = center + post

        if start < 0 or end > len(audio):
            skipped += 1
            continue

        clip = audio[start:end].copy()
        peak = np.max(np.abs(clip))
        if peak > 1e-6:
            clip /= peak   # normalize per-clip to [-1, 1]

        key_dir = output_dir / key
        key_dir.mkdir(exist_ok=True)
        idx = counts.get(key, 0)
        sf.write(str(key_dir / f'{key}_{idx:04d}.wav'), clip, sr)
        counts[key] = idx + 1

    return {
        'clips_saved':   sum(counts.values()),
        'events_total':  len(events),
        'skipped':       skipped,
        'clips_per_key': counts,
    }


# ── Onset detection ───────────────────────────────────────────────────────────

def detect_onsets(audio, sr, delta=0.07, wait_ms=150):
    """
    Onset detection using librosa's onset_detect with tuned keyboard parameters.
    Returns sample-index array of detected onsets.

    Works well on short demo recordings (5-30s of continuous typing).
    The percentile-based approach in the original version fires too often on
    long sparse recordings like guided collection sessions.

    delta:    minimum onset strength above the local mean (tune up to reduce FP)
    wait_ms:  minimum gap between detected onsets in milliseconds
    """
    hop    = 256
    wait_frames = max(1, int(wait_ms / 1000 * sr / hop))

    onset_samples = librosa.onset.onset_detect(
        y=audio, sr=sr,
        hop_length=hop,
        delta=delta,
        wait=wait_frames,
        units='samples',
        backtrack=False,
    )
    return onset_samples.astype(int)


def detect_keystrokes(audio, sr, min_gap_ms=230, rel_thresh=0.35):
    """
    One onset per keypress — for audio-only inference (the demo).

    Every key makes two transients: the push (key bottoms out) and the release
    (key springs back). Raw onset detection fires on both, roughly doubling the
    count and feeding the classifier release clicks it was never trained on.
    Two cleanups, both grounded in measured behaviour:

      1. Non-maximum suppression: within any min_gap window keep only the
         loudest transient. The push is ~4x louder than the release, so the
         push wins and the release is dropped.
      2. Strength floor: drop survivors weaker than rel_thresh x the median
         keystroke strength — removes stray clicks (stream startup, etc.).

    Type with a clear gap between keys (tap, don't hold) for the cleanest count.
    """
    onsets = detect_onsets(audio, sr)
    if len(onsets) == 0:
        return onsets

    env = np.convolve(np.abs(audio), np.ones(48) / 48, mode='same')
    w   = int(0.03 * sr)
    strength = {int(o): float(env[o: min(len(env), o + w)].max()) for o in onsets}

    gap  = int(min_gap_ms / 1000 * sr)
    kept = []
    for o in sorted(strength, key=strength.get, reverse=True):   # strongest first
        if all(abs(o - k) >= gap for k in kept):
            kept.append(o)

    if kept:
        med  = float(np.median([strength[k] for k in kept]))
        kept = [k for k in kept if strength[k] >= rel_thresh * med]

    return np.array(sorted(kept), dtype=int)


# ── Isolation quality metrics ─────────────────────────────────────────────────

def evaluate_isolation(detected, true_events, sr, tolerance_ms=30.0):
    """
    Compare detected onset samples to ground-truth timestamps.
    Reports precision, recall, and F1 — a separate number from classifier accuracy.
    """
    true_samples = np.array([ts * sr for ts, _ in true_events])
    tol = int(tolerance_ms / 1000 * sr)

    matched_gt = set()
    tp = 0

    for d in detected:
        diffs      = np.abs(true_samples - d)
        nearest    = int(np.argmin(diffs))
        if diffs[nearest] < tol and nearest not in matched_gt:
            tp += 1
            matched_gt.add(nearest)

    precision = tp / len(detected)       if len(detected)    > 0 else 0.0
    recall    = tp / len(true_samples)   if len(true_samples) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall + 1e-9)

    return {
        'true_positives':  tp,
        'false_positives': len(detected)    - tp,
        'false_negatives': len(true_samples) - tp,
        'precision':       round(precision, 4),
        'recall':          round(recall,    4),
        'f1':              round(f1,        4),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_labels(labels_path):
    events = []
    with open(labels_path, newline='') as f:
        for row in csv.DictReader(f):
            events.append((float(row['timestamp_sec']), row['key']))
    return events


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    wav_path    = Path(sys.argv[1])
    labels_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    session     = wav_path.stem
    out_dir     = CLIPS_DIR / session

    if labels_path and labels_path.exists():
        print(f'Extracting clips using ground-truth labels from {labels_path}...')
        stats = extract_clips(wav_path, labels_path, out_dir)
        print(json.dumps(stats, indent=2))

        # Bonus: evaluate onset detector against the same ground truth
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio_proc = resample_audio(audio, sr, PROCESS_SR)
        events     = _load_labels(labels_path)
        detected   = detect_onsets(audio_proc, PROCESS_SR)
        iso_stats  = evaluate_isolation(detected, events, PROCESS_SR)
        print('\nOnset detector quality (vs ground truth):')
        print(json.dumps(iso_stats, indent=2))
    else:
        print(f'Running onset detection on {wav_path} (no label file)...')
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio_proc = resample_audio(audio, sr, PROCESS_SR)
        detected   = detect_onsets(audio_proc, PROCESS_SR)
        print(f'Detected {len(detected)} onsets')
        times      = detected / PROCESS_SR
        print(f'First 10 onset times (s): {times[:10].tolist()}')
