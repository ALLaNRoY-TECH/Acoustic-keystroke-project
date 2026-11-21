"""
synth_data.py — Synthetic keystroke generator (no microphone required).

Why this exists
---------------
The real pipeline needs a microphone and a person typing ~900 times before
anything can be trained. That makes the repo impossible to *verify* without
hardware and a long data-collection session. This script fabricates a small,
physically-plausible keystroke dataset so the whole train -> evaluate -> demo
pipeline can be exercised end-to-end on any machine, in seconds.

The clips are NOT recordings of a real keyboard. Each key is given a distinct,
deterministic acoustic "signature" (a pair of damped resonant transients — a
loud push followed by a quieter release — at a key-specific set of resonant
frequencies) plus per-clip randomisation and background noise. That gives the
classifier genuine, learnable class structure, so a successful run proves the
pipeline works; it says nothing about real-world attack accuracy.

Outputs (raw audio only — identical format to realign.py, so every downstream
stage is unchanged):

  data/clips_aligned/<session>/<key>/<key>_NNNN.wav   training clips
  data/raw/<phrase_name>.wav                          a continuous "phrase"
                                                      recording for demo.py

Usage:
  python synth_data.py                       # default: session 'synth01', 30/key
  python synth_data.py --session synth01 --per-key 30
  python synth_data.py --phrase hello        # also write a demo phrase WAV
  python synth_data.py --per-key 40 --phrase secret --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from config import PROCESS_SR, KEYS, KEY_TO_IDX, DATA_DIR, RAW_DIR

# Match the real aligned-clip geometry exactly (see realign.py).
WINDOW_SAMPLES = 4000      # 250 ms at 16 kHz
ONSET_FRAC     = 0.25      # onset sits a quarter of the way into each clip


# ── Per-key acoustic signature ──────────────────────────────────────────────────

def _key_resonances(key):
    """
    Deterministic set of resonant frequencies (Hz) for a key.

    Keyboard energy lives below ~8 kHz. We spread 36 keys across that band and
    give each a small chord of partials so the spectrograms are visually and
    statistically distinct, but neighbouring key indices overlap a little —
    loosely echoing the real result that physically close keys are the most
    confusable.
    """
    idx = KEY_TO_IDX[key]
    base = 450.0 + idx * 190.0          # 450 Hz .. ~7.1 kHz across 36 keys
    return [base, base * 1.9, base * 2.7 + 120.0]


def _damped_click(n, freqs, decay_samples, rng, phase_jitter=True):
    """A sum of exponentially-damped sinusoids: the *resonant* (key-specific)
    part of a keystroke transient."""
    t = np.arange(n)
    env = np.exp(-t / decay_samples)
    sig = np.zeros(n, dtype=np.float64)
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi) if phase_jitter else 0.0
        amp = rng.uniform(0.7, 1.0)
        sig += amp * np.sin(2 * np.pi * f * t / PROCESS_SR + phase)
    return sig * env


def _broadband_burst(n, decay_samples, rng):
    """Short impulsive broadband 'click' — the physical impact energy common to
    every key. Key-INDEPENDENT (carries no class information), but it gives every
    keystroke a strong, uniform onset that onset detectors latch onto, just like
    a real key press. This is what stops high-frequency keys being filtered out
    as 'too weak'."""
    t = np.arange(n)
    return rng.normal(0, 1, n) * np.exp(-t / decay_samples)


def synth_clip(key, rng):
    """
    One WINDOW_SAMPLES-long synthetic keystroke clip for `key`.

    Structure: a loud push transient at the onset position, a quieter release
    transient 60–95 ms later, light broadband noise throughout, and small
    per-clip jitter in timing, amplitude and decay so no two clips are
    identical. Peak-normalised to [-1, 1], exactly like the real clips.
    """
    freqs = _key_resonances(key)
    clip  = rng.normal(0, 0.01, WINDOW_SAMPLES)       # background noise floor

    onset = int(ONSET_FRAC * WINDOW_SAMPLES) + rng.integers(-60, 60)
    onset = int(np.clip(onset, 0, WINDOW_SAMPLES - 400))

    # Push transient (the key bottoming out) — the loud one. It is a sharp
    # broadband click (uniform across keys, drives onset detection) plus a
    # key-specific resonant tail (what the classifier actually learns).
    push_decay = rng.uniform(240, 340)
    push_len   = min(WINDOW_SAMPLES - onset, 1500)
    push_freqs = [f * rng.uniform(0.97, 1.03) for f in freqs]
    burst_decay = rng.uniform(25, 45)                 # ~2 ms broadband impact
    clip[onset:onset + push_len] += 0.9 * _broadband_burst(push_len, burst_decay, rng)
    clip[onset:onset + push_len] += 0.8 * _damped_click(push_len, push_freqs, push_decay, rng)

    # Release transient (the key springing back) — quieter, slightly later.
    rel_gap = onset + int(rng.uniform(0.060, 0.095) * PROCESS_SR)
    if rel_gap < WINDOW_SAMPLES - 200:
        rel_len = min(WINDOW_SAMPLES - rel_gap, 900)
        clip[rel_gap:rel_gap + rel_len] += 0.25 * _broadband_burst(rel_len, burst_decay, rng)
        clip[rel_gap:rel_gap + rel_len] += 0.22 * _damped_click(rel_len, push_freqs, push_decay * 0.6, rng)

    peak = np.max(np.abs(clip))
    if peak > 1e-9:
        clip = clip / peak
    return clip.astype(np.float32)


# ── Dataset generation ──────────────────────────────────────────────────────────

def generate_clips(session, per_key, seed):
    rng = np.random.default_rng(seed)
    out_dir = DATA_DIR / 'clips_aligned' / session
    total = 0
    for key in KEYS:
        key_dir = out_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_key):
            clip = synth_clip(key, rng)
            sf.write(str(key_dir / f'{key}_{i:04d}.wav'), clip, PROCESS_SR)
            total += 1
    print(f'Wrote {total} synthetic clips ({per_key}/key x {len(KEYS)} keys) -> {out_dir}')
    return out_dir


def generate_phrase(text, seed):
    """
    A continuous recording of `text` being 'typed', for demo.py.

    Keys are spaced ~0.32 s apart (well above demo's 230 ms min-gap) with small
    jitter, so onset detection finds one transient per key.
    """
    text = ''.join(c for c in text.lower() if c in KEY_TO_IDX)
    if not text:
        raise ValueError('Phrase has no characters in the a-z0-9 vocabulary.')

    rng = np.random.default_rng(seed + 999)
    gap = int(0.36 * PROCESS_SR)
    pad = int(0.30 * PROCESS_SR)
    total_len = pad * 2 + gap * len(text)
    buf = rng.normal(0, 0.01, total_len).astype(np.float64)

    onset_in_clip = int(ONSET_FRAC * WINDOW_SAMPLES)
    pos = pad
    for ch in text:
        clip = synth_clip(ch, rng)                       # reuse the same signatures
        # Normalise each keystroke by RMS (not peak) so all presses have equal
        # energy regardless of random spikes — otherwise one clip can end up far
        # quieter than the rest and the detector's strength floor drops it.
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms > 1e-9:
            clip = clip / rms * 0.15
        start = pos - onset_in_clip + int(rng.uniform(-0.02, 0.02) * PROCESS_SR)
        start = max(0, start)
        end = min(total_len, start + WINDOW_SAMPLES)
        buf[start:end] += clip[:end - start]
        pos += gap + int(rng.uniform(-0.03, 0.03) * PROCESS_SR)

    peak = np.max(np.abs(buf))
    if peak > 1e-9:
        buf = buf / peak * 0.9
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f'{text}.wav'
    sf.write(str(out), buf.astype(np.float32), PROCESS_SR)
    print(f"Wrote synthetic phrase '{text}' ({len(text)} keys) -> {out}")
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session',  default='synth01', help="clip session name")
    p.add_argument('--per-key',  type=int, default=30, help='clips per key')
    p.add_argument('--phrase',   default=None, help='also write a demo phrase WAV, e.g. --phrase hello')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    generate_clips(args.session, args.per_key, args.seed)
    if args.phrase:
        generate_phrase(args.phrase, args.seed)
    print('\nNext:')
    print('  python train.py    --model baseline --clips data/clips_aligned')
    print('  python evaluate.py --model baseline')
    if args.phrase:
        print(f'  python demo.py data/raw/{"".join(c for c in args.phrase.lower() if c.isalnum())}.wav --model baseline')
