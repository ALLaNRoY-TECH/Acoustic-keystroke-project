from pathlib import Path

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 44100   # Hz — recording rate
PROCESS_SR      = 16000   # Hz — processing rate (matches the 2023 paper)

# Keystroke extraction window around each keydown event
WINDOW_PRE_MS   = 80      # ms before keydown timestamp (covers OS-delivery latency)
WINDOW_POST_MS  = 170     # ms after  keydown timestamp (captures push + release peak)
# Total window = 250 ms = 4000 samples at 16 kHz

# ── Mel-spectrogram ────────────────────────────────────────────────────────────
N_MELS          = 64      # mel frequency bins
N_FFT           = 512     # FFT size  (~32 ms at 16 kHz)
HOP_LENGTH      = 80      # hop size  (  5 ms at 16 kHz → ~44 frames for 250 ms)
# Output after padding: (N_MELS × IMG_WIDTH) = (64 × 64)
IMG_HEIGHT      = 64      # = N_MELS (no resize needed on freq axis)
IMG_WIDTH       = 64      # time frames; short windows are zero-padded to this

# ── Classes ────────────────────────────────────────────────────────────────────
KEYS        = list('abcdefghijklmnopqrstuvwxyz0123456789')
NUM_CLASSES = len(KEYS)           # 36
KEY_TO_IDX  = {k: i for i, k in enumerate(KEYS)}
IDX_TO_KEY  = {i: k for k, i in KEY_TO_IDX.items()}

# ── Data collection ────────────────────────────────────────────────────────────
PRESSES_PER_KEY = 25     # guided-mode target per key → 900 total clips

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 100
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
SEED            = 42

# Augmentation applied during training (train split only).
# The masks are classic SpecAugment; the rest simulate the cross-session
# variation we measured (onset jitter, noise-floor + normalisation differences)
# so the model stops memorising one session's exact fingerprint.
FREQ_MASK_PARAM = 6     # max width of frequency mask
TIME_MASK_PARAM = 6     # max width of time mask
NUM_MASKS       = 2     # number of freq + time masks
AUG_MAX_SHIFT   = 6     # max time-frame roll (onset-position jitter)
AUG_NOISE_STD   = 0.05  # std of additive spectrogram noise (noise-floor jitter)
AUG_CONTRAST    = 0.15  # gain/offset jitter (per-clip normalisation differences)
AUG_PROB        = 0.5   # per-effect probability

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('data')
RAW_DIR     = DATA_DIR / 'raw'      # WAV + CSV files from collect_data.py
CLIPS_DIR   = DATA_DIR / 'clips'    # per-keystroke clips from isolate.py
MODELS_DIR  = Path('saved_models')
RESULTS_DIR = Path('results')
