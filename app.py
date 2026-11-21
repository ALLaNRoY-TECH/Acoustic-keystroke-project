"""
app.py — Live web demo for Acoustic Keystroke Recovery.

A Gradio web app (the one that runs on Hugging Face Spaces) that lets anyone,
with no microphone and no setup, see the trained model work:

  * "Type a word"  -> the app synthesises the sound of that word being typed,
                      plays it back, and the real trained model recovers the text.
  * "Upload audio" -> upload a .wav of keystrokes and the model recovers it.

It reuses the exact inference pipeline from demo.py and the exact synthetic
audio generator from synth_data.py, so what you see here is the same code the
command-line tools run — nothing is faked.

Run locally:
    python app.py
Then open the printed http://127.0.0.1:7860 URL.
"""

import tempfile

# On Hugging Face "ZeroGPU" Spaces, a function must be wrapped with @spaces.GPU or
# the Space fails to start. Locally (no `spaces` package) this is a harmless no-op.
try:
    import spaces

    def gpu(fn):
        # The model needs ~1s, so reserve only 3s of the (tiny) free quota
        # instead of the 60s default — this multiplies runs per visitor.
        return spaces.GPU(duration=3)(fn)
except Exception:
    def gpu(fn):
        return fn

import numpy as np
import soundfile as sf
import gradio as gr

from config import PROCESS_SR, KEY_TO_IDX, IDX_TO_KEY
from realign import WINDOW_SAMPLES, ONSET_FRAC
from synth_data import synth_clip
from demo import load_trained_model, infer_wav
from decode import greedy_decode, correct_word, load_word_list

# ── One-time setup (loaded once when the server starts) ─────────────────────────
MODEL = load_trained_model("baseline")          # pure-PyTorch, no torchvision needed
try:
    WORD_LIST = load_word_list()
except Exception:
    WORD_LIST = []

MAX_LEN = 24                                     # keep demo audio short & snappy


# ── In-memory synthetic "typing" audio (no disk writes, no name collisions) ─────
def synth_phrase_buffer(text, seed=42):
    """Return (sample_rate, float32 waveform) of `text` being typed.

    Identical maths to synth_data.generate_phrase, but returns the buffer in
    memory so many web visitors can run at once without clashing on a filename.
    """
    text = "".join(c for c in text.lower() if c in KEY_TO_IDX)[:MAX_LEN]
    if not text:
        raise ValueError("Please use letters a–z or digits 0–9.")

    rng = np.random.default_rng(seed + 999)
    gap = int(0.36 * PROCESS_SR)
    pad = int(0.30 * PROCESS_SR)
    total_len = pad * 2 + gap * len(text)
    buf = rng.normal(0, 0.01, total_len).astype(np.float64)

    onset_in_clip = int(ONSET_FRAC * WINDOW_SAMPLES)
    pos = pad
    for ch in text:
        clip = synth_clip(ch, rng)
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms > 1e-9:
            clip = clip / rms * 0.15
        start = max(0, pos - onset_in_clip + int(rng.uniform(-0.02, 0.02) * PROCESS_SR))
        end = min(total_len, start + WINDOW_SAMPLES)
        buf[start:end] += clip[:end - start]
        pos += gap + int(rng.uniform(-0.03, 0.03) * PROCESS_SR)

    peak = np.max(np.abs(buf))
    if peak > 1e-9:
        buf = buf / peak * 0.9
    return PROCESS_SR, buf.astype(np.float32)


def _run_model_on_wav(wav_path):
    """Run the shared inference pipeline and format the results for the web UI."""
    probs = infer_wav(wav_path, MODEL, verbose=False)
    if not probs:
        return "—", "No keystrokes were detected in the audio.", ""

    greedy = greedy_decode(probs)

    # Optional dictionary correction (only meaningful for a single real word).
    corrected = ""
    if WORD_LIST:
        try:
            corrected = correct_word(probs, WORD_LIST)
        except Exception:
            corrected = ""

    # Per-keystroke confidence table.
    rows = ["| # | Predicted key | Confidence |", "|---|---|---|"]
    for i, p in enumerate(probs, 1):
        k = IDX_TO_KEY[int(np.argmax(p))]
        rows.append(f"| {i} | `{k}` | {float(np.max(p)) * 100:.0f}% |")
    table = "\n".join(rows)

    headline = greedy
    detail = f"**Detected {len(probs)} keystrokes.**\n\n"
    detail += f"**Raw model output:** `{greedy}`\n\n"
    if corrected and corrected != greedy:
        detail += f"**After dictionary correction:** `{corrected}`\n\n"
        headline = corrected
    return headline, detail, table


# ── Tab 1: type a word ──────────────────────────────────────────────────────────
@gpu
def demo_typed(word):
    try:
        sr, buf = synth_phrase_buffer(word)
    except ValueError as e:
        return None, "—", str(e), ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, buf, sr)
        wav_path = f.name
    headline, detail, table = _run_model_on_wav(wav_path)
    return (sr, buf), headline, detail, table


# ── Tab 2: upload audio ─────────────────────────────────────────────────────────
@gpu
def demo_uploaded(audio_path):
    if not audio_path:
        return "—", "Please upload or record a .wav file first.", ""
    return _run_model_on_wav(audio_path)


# ── Interface ───────────────────────────────────────────────────────────────────
INTRO = """
# 🎧 → ⌨️  Acoustic Keystroke Recovery — Live Demo

This is a working machine-learning model that **reads typed text back from the
*sound* of the keys**. Type a word below: the app makes the sound of it being
typed, then the model listens and recovers the letters — live, in your browser.

> Built as a reproduction of published research (Harrison et al., 2023). The audio
> here is **synthetic** so it works with no microphone; accuracy on real recordings
> depends on the keyboard and room. For **authorised, ethical use only.**
"""

with gr.Blocks(title="Acoustic Keystroke Recovery") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("⌨️  Type a word"):
        gr.Markdown("Type a word (letters/digits), press **Recover**, and hear + see the model work.")
        with gr.Row():
            word_in = gr.Textbox(label="Word to type", value="hello", max_lines=1)
            run_btn = gr.Button("Recover 🔍", variant="primary")
        gr.Examples(["hello", "secret", "password", "banana", "coffee", "7key"], word_in)
        audio_out = gr.Audio(label="🔊 The 'typing' sound the model hears", type="numpy")
        result_big = gr.Textbox(label="✅ Recovered text", interactive=False)
        detail_md = gr.Markdown()
        table_md = gr.Markdown()
        run_btn.click(demo_typed, word_in, [audio_out, result_big, detail_md, table_md])
        word_in.submit(demo_typed, word_in, [audio_out, result_big, detail_md, table_md])

    with gr.Tab("🎙️  Upload / record audio"):
        gr.Markdown(
            "Upload a `.wav` of keystrokes (or record one). Space the keys ~0.3s apart. "
            "Real-world accuracy varies by keyboard and microphone."
        )
        audio_in = gr.Audio(label="Keystroke audio (.wav)", type="filepath", sources=["upload", "microphone"])
        up_btn = gr.Button("Recover 🔍", variant="primary")
        up_big = gr.Textbox(label="✅ Recovered text", interactive=False)
        up_detail = gr.Markdown()
        up_table = gr.Markdown()
        up_btn.click(demo_uploaded, audio_in, [up_big, up_detail, up_table])

    gr.Markdown(
        "---\n*Model: lightweight CNN over per-keystroke mel-spectrograms · "
        "92% per-key accuracy on the synthetic test set. "
        "Code & full write-up on [GitHub](https://github.com/ALLaNRoY-TECH/Acoustic-keystroke-project).*"
    )

if __name__ == "__main__":
    demo.launch()
