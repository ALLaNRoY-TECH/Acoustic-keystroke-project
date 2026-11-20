"""
inspect_snr.py — is the recording good enough, or is it too noisy to learn from?

Checks whether the keystroke clicks actually stand above the noise floor, on the
ORIGINAL recording:

  * absolute signal levels (was the mic gain too low?)
  * per-keystroke SNR  (peak of the click vs the room noise floor, in dB)
  * whether a dominant hum/tone is swamping everything

Saves results/diag_waveform.png: one keystroke region with the click marked.

Usage:
  python inspect_snr.py                                   # defaults to session01
  python inspect_snr.py data/raw/session02.wav data/raw/session02.csv
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

from isolate import resample_audio, _load_labels
from config import PROCESS_SR, RESULTS_DIR

# Clicks sit ~113 ms before the wall-clock label (see realign.py).
SHIFT = -0.113


def inspect(wav_path, csv_path):
    audio, sr0 = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f'Raw: {len(audio)} samples @ {sr0} Hz = {len(audio)/sr0:.1f}s')
    print(f'  abs-peak {np.max(np.abs(audio)):.4f}   RMS {np.sqrt(np.mean(audio**2)):.4f}')
    print('  (peak near 1.0 = healthy gain; peak <0.1 = mic gain was too low)')

    a = resample_audio(audio, sr0, PROCESS_SR)
    sr = PROCESS_SR

    win = 48
    env = np.convolve(np.abs(a), np.ones(win) / win, mode='same')
    floor = np.percentile(env, 20)
    print(f'\nNoise floor (20th pct of envelope): {floor:.5f}')

    events = _load_labels(csv_path)
    snrs = []
    for t, key in events[:200]:
        c = int((t + SHIFT) * sr)
        lo, hi = max(0, c - int(0.04 * sr)), min(len(a), c + int(0.04 * sr))
        if hi - lo < 10:
            continue
        peak = env[lo:hi].max()
        snrs.append(20 * np.log10((peak + 1e-9) / (floor + 1e-9)))

    if snrs:
        snrs = np.array(snrs)
        print('\nPer-keystroke SNR (click peak vs noise floor):')
        print(f'  mean {snrs.mean():.1f} dB   median {np.median(snrs):.1f} dB   '
              f'min {snrs.min():.1f}   max {snrs.max():.1f}')
        print('  Rule of thumb:  >20 dB great,  12-20 dB workable,  <10 dB very hard')
    else:
        print('\nNo usable keystroke windows found in the label file.')

    spec = np.abs(np.fft.rfft(a[:sr * 5] * np.hanning(min(len(a), sr * 5))))
    freqs = np.fft.rfftfreq(min(len(a), sr * 5), 1 / sr)
    top = np.argsort(spec)[-5:][::-1]
    print('\nLoudest frequency components in first 5s:')
    for k in top:
        print(f'  {freqs[k]:6.0f} Hz  (relative {spec[k]/np.median(spec):.0f}x median)')

    if events:
        i = min(10, len(events) - 1)
        t0 = events[i][0] + SHIFT
        c = int(t0 * sr)
        lo, hi = max(0, c - int(0.15 * sr)), min(len(a), c + int(0.15 * sr))
        seg = a[lo:hi]
        fig, ax = plt.subplots(figsize=(9, 3))
        ms = np.arange(len(seg)) / sr * 1000
        ax.plot(ms, seg, lw=0.5, color='steelblue')
        ax.plot(ms, env[lo:hi], color='red', lw=1, label='envelope')
        ax.axhline(floor, color='gray', ls='--', lw=0.8, label='noise floor')
        ax.set_xlabel('ms'); ax.set_ylabel('amplitude')
        ax.set_title(f"One keystroke region (key '{events[i][1]}')")
        ax.legend(); plt.tight_layout()
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / 'diag_waveform.png'
        plt.savefig(out, dpi=130); plt.close()
        print(f'\nWaveform -> {out}')


if __name__ == '__main__':
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/raw/session01.wav')
    csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/raw/session01.csv')
    if not wav.exists() or not csv.exists():
        print(f'Need a recording + labels. Missing: '
              f'{"" if wav.exists() else wav} {"" if csv.exists() else csv}'.strip())
        print('Record one first:  python collect_data.py guided session01')
        sys.exit(1)
    inspect(wav, csv)
