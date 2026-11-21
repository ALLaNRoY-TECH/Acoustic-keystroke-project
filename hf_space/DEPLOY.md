# Deploying the live demo to Hugging Face Spaces (free, ~5 minutes)

This folder contains everything needed to host the interactive model online.
Follow these steps once — after that, the demo stays live 24/7 at your own URL.

## Steps

1. **Make a free account** at https://huggingface.co/join (just email + password).

2. **Create a Space:** go to https://huggingface.co/new-space
   - **Owner:** your username
   - **Space name:** `keystroke-recovery`  (or anything you like)
   - **License:** MIT
   - **SDK:** choose **Gradio**
   - **Hardware:** **CPU basic** (free)
   - Visibility: **Public**
   - Click **Create Space**.

3. **Upload the files:** on your new Space page, click **Files** → **+ Add file**
   → **Upload files**. Drag in **everything inside this `hf_space` folder**:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `config.py`, `realign.py`, `synth_data.py`, `demo.py`, `decode.py`,
     `dataset.py`, `isolate.py`, `models.py`
   - the `saved_models` folder (with `best_baseline.pt` inside)

   Then click **Commit changes to main**.

4. **Wait ~2–4 minutes** while it says *Building*. When it turns to *Running*,
   your live demo is online.

5. **Your demo URL** will be:
   `https://huggingface.co/spaces/<your-username>/keystroke-recovery`

   The **embed URL** (for the website) is:
   `https://<your-username>-keystroke-recovery.hf.space`

6. **Tell me that URL** and I'll plug it into your website so the live demo
   appears right inside your GitHub page.

## Notes
- First load after inactivity can take ~30s while the Space wakes up — normal on
  the free tier.
- If a file is missing, the build log (Logs tab) will say which one; just upload it.
