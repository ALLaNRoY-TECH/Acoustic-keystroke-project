"""
Stage A — Data collection.

Records microphone audio while logging keypress timestamps.

  meter                 Live input-level meter. RUN THIS FIRST and raise your
                        mic gain until typing peaks land in the GOOD zone.
  guided   <name>       Prompts you key-by-key to build a balanced 36-class set.
  free     <name>       Records everything you type; for natural-text sessions.

Usage:
  python collect_data.py meter
  python collect_data.py guided session02
  python collect_data.py free   session03

Output (guided/free):
  data/raw/<name>.wav   continuous audio
  data/raw/<name>.csv   timestamp_sec, key

Why the meter matters:
  The first dataset was recorded at ~-46 dBFS (peak 0.0047) — the keystroke
  clicks were only ~4 dB above the noise floor and nothing could be learned.
  Keystrokes need to be captured well above the floor; software gain can't fix
  a quiet recording because it amplifies the noise equally. Get the gain right
  at the source using the meter before collecting a full session.
"""

import csv
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard as kb

from config import SAMPLE_RATE, PRESSES_PER_KEY, KEYS, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)

# A recording whose peak never exceeds this is almost certainly too quiet to use.
MIN_PEAK = 0.03          # ~-30 dBFS
GOOD_LO, GOOD_HI = 0.05, 0.85


def _dbfs(x):
    return 20 * np.log10(x + 1e-9)


# ── Live level meter (pre-flight) ───────────────────────────────────────────────

def level_meter():
    """Show a live peak meter so the user can set mic gain before recording."""
    _list_devices()
    print('\nLive mic meter — type on your keyboard and watch the bar.')
    print(f'Target: peaks in the GOOD zone ({GOOD_LO:.2f}-{GOOD_HI:.2f}). '
          'Press Enter to stop.\n')

    latest = {'peak': 0.0, 'over': False}

    def cb(indata, frames, t_info, status):
        latest['peak'] = float(np.max(np.abs(indata[:, 0])))
        if status:
            latest['over'] = True

    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()

    peak_hold = 0.0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        callback=cb, blocksize=1024):
        while not stop.is_set():
            p = latest['peak']
            peak_hold = max(peak_hold * 0.97, p)
            n = 40
            fill = int(min(1.0, p / GOOD_HI) * n)
            if peak_hold < MIN_PEAK:
                zone = 'TOO LOW — raise mic gain / boost'
            elif peak_hold > 0.97:
                zone = 'CLIPPING — lower gain a touch'
            else:
                zone = 'GOOD'
            print(f'\r[{"#" * fill}{" " * (n - fill)}] '
                  f'peak {p:5.3f} ({_dbfs(p):6.1f} dBFS)  hold {peak_hold:5.3f}  {zone}     ',
                  end='', flush=True)
            time.sleep(0.05)

    print('\n')
    if peak_hold < MIN_PEAK:
        print('Mic level is still too low. In Windows: Settings > System > Sound >')
        print('  your input device > set Volume high, and enable Microphone Boost if')
        print('  available (mmsys.cpl > Recording > Properties > Levels). Move the mic')
        print('  closer to the keyboard, then run the meter again.')
    else:
        print('Level looks usable. You can now run a guided session.')


def _list_devices():
    print('Input devices:')
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            mark = '  <- default' if i == default_in else ''
            print(f'  [{i}] {d["name"]}{mark}')


# ── Recorder ────────────────────────────────────────────────────────────────────

class _Recorder:
    """Thread-safe audio buffer. Records t=0 at the FIRST real audio block so
    keystroke timestamps line up with the audio sample clock (the old code set
    t=0 before the device started delivering audio, causing a ~113 ms offset)."""

    def __init__(self):
        self._chunks = []
        self._lock = threading.Lock()
        self.start_perf = None        # perf_counter at first audio sample
        self.overflowed = False

    def callback(self, indata, frames, time_info, status):
        if status:
            self.overflowed = True
        with self._lock:
            if self.start_perf is None:
                self.start_perf = time.perf_counter()
            self._chunks.append(indata.copy())

    def get_audio(self):
        with self._lock:
            return (np.concatenate(self._chunks, axis=0)
                    if self._chunks else np.zeros((0, 1), dtype=np.float32))


def _preflight_gain_check():
    """Capture ~1.5 s and refuse to waste a session on a too-quiet mic."""
    print('Checking mic level (stay quiet for ~1.5 s)...')
    buf = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=1, dtype='float32')
    sd.wait()
    peak = float(np.max(np.abs(buf)))
    print(f'  ambient peak {peak:.4f} ({_dbfs(peak):.1f} dBFS)')
    # Now ask for one test tap so we see a keystroke level too.
    print('  Tap the SPACE bar a few times now...')
    buf2 = sd.rec(int(2.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                  channels=1, dtype='float32')
    sd.wait()
    kpeak = float(np.max(np.abs(buf2)))
    print(f'  keystroke peak {kpeak:.4f} ({_dbfs(kpeak):.1f} dBFS)')
    if kpeak < MIN_PEAK:
        print(f'\n  WARNING: keystroke peak {kpeak:.4f} is below {MIN_PEAK} — the last')
        print('  dataset failed for exactly this reason. Run `python collect_data.py meter`')
        print('  and raise your mic gain first.')
        resp = input('  Record anyway? [y/N] ').strip().lower()
        return resp == 'y'
    print('  Level OK.\n')
    return True


# ── Session ─────────────────────────────────────────────────────────────────────

def collect_session(session_name, mode='guided'):
    if not _preflight_gain_check():
        print('Aborted. Fix the mic level and try again.')
        return

    recorder = _Recorder()
    events   = []
    done     = threading.Event()

    def on_press(key):
        if recorder.start_perf is None:     # audio not flowing yet
            return
        try:
            ch = key.char
        except AttributeError:
            return
        if ch and ch.lower() in KEYS:
            events.append((time.perf_counter() - recorder.start_perf, ch.lower()))

    def on_release(key):
        if key == kb.Key.esc:
            done.set()
            return False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype='float32',
        callback=recorder.callback, blocksize=2048, latency='high',
    )

    with stream:
        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # Wait until the first audio block has actually arrived before prompting,
        # so no early keystrokes are logged against a null time reference.
        while recorder.start_perf is None:
            time.sleep(0.005)

        if mode == 'guided':
            _guided_loop(events, done)
        else:
            print('\nFree typing mode — press Esc to finish.\n')
            done.wait()

        listener.stop()

    if recorder.overflowed:
        print('\nNote: audio buffer overflowed at least once (some samples dropped).')
        print('isolate.py / realign.py snap clips onto the detected onset, so small')
        print('drift is corrected — but close other heavy apps for the cleanest capture.')

    _save(session_name, recorder.get_audio(), events)


def _guided_loop(events, done):
    print(f'\nGuided mode — press each key {PRESSES_PER_KEY} times.')
    print('Press Enter before each key when prompted, then type that key repeatedly.\n')

    for target_key in KEYS:
        if done.is_set():
            break

        print(f"─── Key '{target_key}' ───")
        print('Press Enter when ready...', end='', flush=True)
        try:
            sys.stdin.readline()
        except KeyboardInterrupt:
            break

        before = len(events)
        print(f"Type '{target_key}' {PRESSES_PER_KEY} times:")

        while not done.is_set():
            count = sum(1 for _, k in events[before:] if k == target_key)
            print(f'\r  {count}/{PRESSES_PER_KEY}', end='', flush=True)
            if count >= PRESSES_PER_KEY:
                print()
                break
            time.sleep(0.05)

    done.set()


def _save(session_name, audio, events):
    wav_path = RAW_DIR / f'{session_name}.wav'
    csv_path = RAW_DIR / f'{session_name}.csv'

    sf.write(str(wav_path), audio, SAMPLE_RATE)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_sec', 'key'])
        writer.writerows(events)

    counts   = Counter(k for _, k in events)
    duration = len(audio) / SAMPLE_RATE
    peak     = float(np.max(np.abs(audio))) if len(audio) else 0.0

    print(f"\nSaved {len(events)} keystrokes  |  {duration:.1f}s of audio")
    print(f"  peak amplitude {peak:.4f} ({_dbfs(peak):.1f} dBFS)"
          f"{'   <-- TOO LOW, recording likely unusable' if peak < MIN_PEAK else ''}")
    print(f"  WAV    -> {wav_path}")
    print(f"  Labels -> {csv_path}")
    print('\nPresses per key:')
    for k in sorted(counts):
        bar = '#' * counts[k]
        print(f'  {k}: {counts[k]:3d}  {bar}')

    missing = [k for k in KEYS if k not in counts]
    if missing:
        print(f'\nKeys with no data: {missing}')
        print('Run another guided session to fill gaps.')

    print('\nNext: python realign.py '
          f'{wav_path.as_posix()} {csv_path.as_posix()}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] not in ('guided', 'free', 'meter'):
        print(__doc__)
        sys.exit(1)

    if args[0] == 'meter':
        level_meter()
    else:
        mode = args[0]
        name = args[1] if len(args) > 1 else datetime.now().strftime('%Y%m%d_%H%M%S')
        collect_session(name, mode)
