# ThereSyn — Completion Guide

Step-by-step walkthrough to finish the remaining Phase 4 tasks.

---

## Task 1: End-to-End Testing with Real Hardware

### Prerequisites
- Webcam connected / built-in camera available
- Audio output working (speakers or headphones)
- All dependencies installed: `pip install -r requirements.txt`

### Step 1 — Headless Smoke Test
Verify the pipeline logic works without hardware:
```bash
python headless_smoke.py
```
**Expected:** Prints `Headless smoke-run OK` and exits.

If this fails, something is broken in the pipeline — fix before proceeding.

### Step 2 — Launch with Webcam + Audio
```bash
PYTHONUNBUFFERED=1 python main.py
```

**Checklist (go through each one):**

| # | What to Check | How | Pass? |
|---|--------------|-----|-------|
| 1 | Webcam feed appears | Window shows your camera image | ☐ |
| 2 | Hand landmarks drawn | Hold hand up — blue/green dots + connections visible | ☐ |
| 3 | FPS counter visible | Top-right shows FPS (~20-30 is fine) | ☐ |
| 4 | Waveform label visible | Shows "Wave: sine" top-right area | ☐ |
| 5 | Pitch responds to X | Pinch + move hand left/right — pitch glides low↔high | ☐ |
| 6 | Volume responds to Y | Pinch + move hand up/down — volume changes | ☐ |
| 7 | Pinch engages sound | Pinch thumb+index → sound starts | ☐ |
| 8 | Release stops sound | Release pinch → sound stops (with ADSR release) | ☐ |
| 9 | Timbre from finger spread | Spread fingers open → brighter tone, close → darker | ☐ |
| 10 | Waveform cycling works | Press `W` → label changes (sine→saw→square→tri), sound changes | ☐ |
| 11 | Visualizer shows waveform | Bottom strip shows waveform animation | ☐ |
| 12 | Latency feels good | Gesture to sound delay should feel <100ms | ☐ |
| 13 | ESC quits cleanly | Press ESC → window closes, no Python crash | ☐ |
| 14 | Camera disconnect recovery | Cover camera briefly — app should tolerate (30-frame grace) | ☐ |

### Step 3 — If Something Fails
- **No sound:** Check system volume, try headphones, check `config.py` SAMPLE_RATE/BUFFER_SIZE
- **Landmarks not tracking:** Ensure good lighting, hand clearly visible, try `HAND_DETECTION_CONFIDENCE = 0.5` in config
- **Stuttering audio:** Increase `BUFFER_SIZE` to 1024 in config, close other apps
- **Camera won't open:** Close Zoom/browser tabs using camera, verify with `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Step 4 — MIDI Test (Optional)
If you have a DAW or virtual MIDI port:
1. Set up a virtual MIDI port (e.g., `loopMIDI` on Windows, `IAC Driver` on macOS)
2. In your DAW, arm a MIDI track listening on that port
3. Run `python main.py` — theremin pitch_bend + CC messages should reach the DAW
4. Verify bend range matches `config.py` → `MIDI_PITCH_BEND_RANGE_SEMITONES` (default 2)

---

## Task 2: Collect Real Gesture Samples

The model needs real hand data for the 7 gesture labels defined in `config.py`:
```
["piano_rest", "theremin_engage", "octave_up", "octave_down",
 "instrument_switch", "play", "stop"]
```

### Step 1 — Launch Collector
```bash
python train_model.py --collect
```

### Step 2 — For Each Gesture Label

You'll see your webcam feed with a status bar at the top.

| Key | Gesture | What to Do With Your Hand |
|-----|---------|--------------------------|
| `1` | piano_rest | Open flat hand, relaxed, fingers slightly apart |
| `2` | theremin_engage | Pinch thumb + index together (like holding something small) |
| `3` | octave_up | Point index finger up |
| `4` | octave_down | Point index finger down |
| `5` | instrument_switch | Peace sign / two fingers up |
| `6` | play | Thumbs up |
| `7` | stop | Closed fist |

### Step 3 — Recording Workflow

For **each** gesture, do this:
1. Press the number key to set the label (e.g., `1` for piano_rest)
2. Make the gesture clearly with your hand visible to the camera
3. Press `SPACE` to record one sample
4. **Vary it slightly** — move your hand to different positions in the frame, angle slightly differently
5. Aim for **at least 20 samples per gesture** (more = better, 30+ is ideal)
6. Repeat for all 7 gestures

**Tips:**
- Good lighting helps a lot
- Keep your full hand in frame
- Try to record in similar conditions to how you'll actually use it
- The status bar shows current label + total sample count
- Don't rush between gestures — the smoother tracks continuously

### Step 4 — Save
Press `S` to save samples to `assets/samples.npz`.

Press `Q` or `ESC` to exit.

**If you want to add more samples later**, just run `--collect` again — but note it starts fresh each time. If you need to accumulate across sessions, you'll need to merge `.npz` files manually.

---

## Task 3: Train Model with Real Data

### Step 1 — Train
```bash
python train_model.py --train --samples assets/samples.npz
```

**Expected output:**
```
[Train] Found classes: ['piano_rest', 'theremin_engage', ...]
[Train] Total: 140 samples, 7 classes, 63 features
[Train] Training accuracy: 0.95+
[Train] Test accuracy:     0.85+
[Train] ONNX export verified — predictions match sklearn
```

**What to look for:**
- Test accuracy > 80% — if lower, collect more samples or ensure gestures are distinct
- Per-class precision/recall in the classification report — watch for low scores on specific gestures
- "WARNING: Class X has only Y samples" → go back and collect more for that gesture

### Step 2 — Verify
```bash
python train_model.py --verify
```

**Expected:**
```
[Verify] Model loaded successfully.
[Verify] Labels: ['piano_rest', 'theremin_engage', ...]
[Verify] Dummy input → piano_rest (conf=0.XX)
[Verify] OK
```

### Step 3 — Real-world Test
Launch the app and test ML gestures work:
```bash
python main.py
```

Try each gesture and check:
- `theremin_engage` → pinch engages sound (same as manual pinch)
- `stop` → disengages sound
- `octave_up` / `octave_down` → pitch jumps octave
- Console prints `[ML] Loaded model from assets/gesture_classifier.onnx` at startup

If confidence is too low / too many false positives, adjust `GESTURE_CONFIDENCE_THRESHOLD` in `config.py` (default 0.6 — raise to 0.7 or 0.8 for stricter detection).

---

## Task 4: Demo Video Recording

### Step 1 — Record with OBS
1. Open OBS Studio
2. Add source: **Window Capture** → select "ThereSyn — AR Theremin-Synthesizer"
3. Add source: **Audio Output Capture** → capture system audio (so theremin sound is included)
4. Hit **Start Recording**
5. Demo the app for 15-30 seconds:
   - Show hand clearly entering frame
   - Pinch to engage → play a simple melody by moving hand
   - Show volume control (up/down)
   - Cycle waveforms with `W`
   - Show finger spread for timbre change
6. Hit **Stop Recording**

### Step 2 — Export to MP4
OBS saves as `.mkv` by default. Convert:
```bash
ffmpeg -i demo.mkv -c:v libx264 -c:a aac -movflags +faststart demo.mp4
```

### Step 3 — Create GIF (for README later)
```bash
# Trim to best 8-12 seconds first
ffmpeg -i demo.mp4 -ss 0:00 -t 0:10 -vf "fps=15,scale=800:-1:flags=lanczos" -y demo-15fps.gif

# Optimize
gifsicle -O3 --colors 256 demo-15fps.gif -o demo-optimized.gif
```

### Step 4 — Save Files
Keep both:
- `demo.mp4` — full quality, for GitHub releases / sharing
- `demo-optimized.gif` — for README (can add later)

---

## Quick Reference — All Commands in Order

```bash
# 1. Headless check
python headless_smoke.py

# 2. Real hardware test
python main.py

# 3. Collect gesture samples (7 gestures × 20+ samples each)
python train_model.py --collect

# 4. Train model
python train_model.py --train --samples assets/samples.npz

# 5. Verify model
python train_model.py --verify

# 6. Test with ML gestures
python main.py

# 7. Record demo (OBS, then convert)
ffmpeg -i demo.mkv -c:v libx264 -c:a aac -movflags +faststart demo.mp4
```

---

## Post-Completion

After all 4 tasks are done:
- [ ] Push final model (`assets/gesture_classifier.onnx`) to GitHub
- [ ] Update `BUILD_PLAN.md` — mark Phase 4 as ✅ Complete
- [ ] Add screenshots/demo video link to README
- [ ] Tag a release: `git tag v1.0.0`
