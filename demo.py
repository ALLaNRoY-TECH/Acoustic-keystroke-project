"""
Demo — Recover typed text from a WAV recording.

Runs the full inference pipeline:
  audio file -> onset detection -> per-keystroke mel-spectrogram -> classifier
  -> (optionally) language-model correction -> recovered text

Usage:
  python demo.py <audio.wav> [--model baseline|coatnet] [--lm] [--word-lengths 5 3 7]

  --lm              apply dictionary word-correction (requires knowing word lengths)
  --word-lengths    space-separated ints; if provided, each segment is corrected
                    independently (e.g. --word-lengths 5 3  for a 5-letter + 3-letter phrase)

Example (full pipeline demo for the video):
  # 1. Record yourself typing a phrase:
  python collect_data.py free demo_phrase
  # 2. Run inference:
  python demo.py data/raw/demo_phrase.wav --model coatnet --lm --word-lengths 8 4 5
"""

import argparse
import sys
import time
import numpy as np
import soundfile as sf
import torch

from config import PROCESS_SR, MODELS_DIR, NUM_CLASSES, IDX_TO_KEY
from isolate import detect_keystrokes, resample_audio
from dataset import audio_to_melspec
from models import build_model
from realign import envelope, find_onset, WINDOW_SAMPLES, ONSET_FRAC
from decode import greedy_decode, correct_word, load_word_list


# ── Model loading ─────────────────────────────────────────────────────────────

def load_trained_model(model_type='baseline'):
    ckpt_path = MODELS_DIR / f'best_{model_type}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'No checkpoint at {ckpt_path}. Run train.py --model {model_type} first.'
        )
    ckpt   = torch.load(ckpt_path, map_location='cpu')
    model  = build_model(model_type, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_wav(wav_path, model, verbose=True):
    """
    Load a WAV, detect onsets, classify each keystroke.
    Returns list of per-class probability arrays in temporal order.

    Clips are built IDENTICALLY to the training data (realign.py): each
    coarse onset is refined to the local envelope onset, then a fixed
    WINDOW_SAMPLES window is taken with the onset at ONSET_FRAC. Matching
    the training extraction is what lets the demo hit the test-set accuracy.
    """
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = resample_audio(audio.astype(np.float32), sr, PROCESS_SR)
    env = envelope(audio)

    coarse = detect_keystrokes(audio, PROCESS_SR)
    if verbose:
        print(f'Detected {len(coarse)} keystrokes in {len(audio)/PROCESS_SR:.2f}s of audio')

    pre    = int(ONSET_FRAC * WINDOW_SAMPLES)
    refine = int(0.03 * PROCESS_SR)        # ±30 ms search to snap onto the onset

    all_probs = []
    for o in coarse:
        lo, hi = max(0, o - refine), min(len(audio), o + refine)
        onset  = find_onset(env, lo, hi)[0] if hi - lo > 10 else o

        start = onset - pre
        end   = start + WINDOW_SAMPLES
        if start < 0 or end > len(audio):
            continue
        clip = audio[start:end].copy()
        peak = np.max(np.abs(clip))
        if peak > 1e-6:
            clip /= peak

        mel = audio_to_melspec(clip)   # (1, H, W)
        x   = torch.from_numpy(mel).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0).numpy()
        all_probs.append(probs)

    return all_probs


# ── Pretty output ─────────────────────────────────────────────────────────────

def _print_per_key(probs_list):
    print('\nPer-keystroke predictions (top-3):')
    for i, probs in enumerate(probs_list):
        top3_idx = np.argsort(probs)[::-1][:3]
        top3     = [(IDX_TO_KEY[j], probs[j]) for j in top3_idx]
        bar      = f'{top3[0][0]} (p={top3[0][1]:.2f})'
        alts     = ', '.join(f'{k}({p:.2f})' for k, p in top3[1:])
        print(f'  [{i+1:3d}] {bar}  alt: {alts}')


def _type_out(text, delay=0.07):
    """Animated character-by-character print — good for video demos."""
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def demo(wav_path, model_type='baseline', use_lm=False, word_lengths=None, verbose=True):
    model      = load_trained_model(model_type)
    probs_list = infer_wav(wav_path, model, verbose=verbose)

    if not probs_list:
        print('No keystrokes detected.')
        return

    if verbose:
        _print_per_key(probs_list)

    greedy = greedy_decode(probs_list)
    print(f'\nGreedy prediction :  ', end='')
    _type_out(greedy)

    if use_lm:
        word_list = load_word_list()
        if word_lengths:
            # Correct each word-length segment independently
            idx, corrected_words = 0, []
            for wlen in word_lengths:
                seg  = probs_list[idx: idx + wlen]
                word = correct_word(seg, word_list) if seg else ''
                corrected_words.append(word)
                idx += wlen
            corrected = ' '.join(corrected_words)
        else:
            # No word-length hints: just correct the whole sequence as one word
            corrected = correct_word(probs_list, word_list)

        print(f'LM-corrected      :  ', end='')
        _type_out(corrected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('wav',           help='Path to WAV file')
    p.add_argument('--model',       default='coatnet', choices=['baseline', 'coatnet'])
    p.add_argument('--lm',          action='store_true', help='Apply LM word correction')
    p.add_argument('--word-lengths', nargs='+', type=int, default=None,
                   help='Known word lengths (e.g. 5 3) for segmented LM correction')
    p.add_argument('--quiet',       action='store_true', help='Skip per-key verbose output')
    args = p.parse_args()

    demo(
        args.wav,
        model_type=args.model,
        use_lm=args.lm,
        word_lengths=args.word_lengths,
        verbose=not args.quiet,
    )
