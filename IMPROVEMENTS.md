# What changed in this version

A log of the fixes and additions made on top of the original pipeline, so the
changes are easy to review and explain.

## New

- **`synth_data.py` — no-microphone synthetic dataset generator.**
  The original repo could not be run at all without first recording ~900 real
  keystrokes. This fabricates physically-plausible keystroke clips (a uniform
  broadband impact "click" that drives onset detection, plus a per-key resonant
  tail that the classifier learns) and a continuous demo phrase, so the whole
  `train → evaluate → demo` pipeline runs end-to-end on any machine in seconds.
  These are **not real recordings** — they verify the pipeline is correct, not
  that the attack works in the wild.

## Fixed

- **Runs on Python 3.14.** `timm` (pretrained CoAtNet) imports `torchvision`,
  which has no compiled 3.14 wheel yet — importing it crashed even the baseline.
  `models.py` now imports timm *lazily* and, when it is unavailable, falls back
  to **`NativeCoAtNet`**: a self-contained conv-plus-transformer network in pure
  PyTorch (no timm, no torchvision, no pretrained weights). Both `--model
  baseline` and `--model coatnet` now run anywhere torch runs.

- **End-to-end demo was dropping the first keystroke** (recovered `ello`
  instead of `hello`). The onset detector's strength filter discarded quiet
  presses; the synthetic keystrokes were narrowband tones whose energy the
  envelope smoothing washed out. Giving every keystroke a realistic broadband
  click (and RMS-normalising the demo phrase) makes all presses detectable.
  Demo now recovers the full string.

- **`decode.py` dictionary works offline.** The nltk word list download could
  fail (no network) and crash the LM step. It now merges the built-in common
  words with the nltk corpus when present, and falls back cleanly otherwise, so
  common words like `hello` are always recoverable.

- **`inspect_snr.py` no longer crashes on import.** It ran file-reading code at
  module top level with a hard-coded `session01` path. Now it takes a CLI
  argument, is guarded behind `__main__`, and exits with a helpful message when
  no recording exists.

## Environment notes

- **Windows long-path (`WinError 206`).** PyTorch's DLLs fail to load from very
  deep paths. Work from a short path, or point a junction at the project
  (`mklink /J C:\dev\akr "<full path>"`) and run from there.
- `requirements.txt` now installs torch from the official index and marks
  `timm`/`torchvision` optional (only needed for the pretrained CoAtNet on
  Python ≤3.13).

## Measured on the synthetic data

| Model                 | Test top-1 | Test top-5 |
|-----------------------|-----------|-----------|
| BaselineCNN           | ~89%      | 100%      |
| NativeCoAtNet         | ~87–90%   | 100%      |

End-to-end demo recovers the typed phrase (`hello`) from audio alone, greedy and
LM-corrected. These numbers describe synthetic data and are a pipeline sanity
check, not a real-world attack result — for that, record your own typing with
`collect_data.py` (see README).
