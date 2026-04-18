# ThereSyn
### AR Theremin-Synthesizer — Play Music in Free Space

ThereSyn is a real-time gesture-based theremin that uses your hands to control pitch, volume, and timbre through a webcam — no physical contact required.

Built with MediaPipe, Pygame, NumPy, and real-time DSP.

<!-- TODO: Add demo GIF here when ready -->
<!-- ![ThereSyn Demo](assets/demo.gif) -->

## ✨ Features

- **🔮 Continuous Pitch Control** — Move your hand left/right to glide across frequencies (C3–C6) with exponential musical mapping
- **🔊 Amplitude Control** — Move your hand up/down to control volume in real-time
- **✋ Pinch to Engage** — Pinch thumb + index to start/stop sound (no accidental triggers)
- **🎛️ Timbre Control** — Spread your fingers open for bright tone, close for warm/dark tone
- **🤖 ML Gesture Classifier** — Lightweight neural network for intelligent gesture detection
- **🎵 MIDI Output** — Stream pitch_bend + CC messages to any DAW (Ableton, FL Studio, etc.)
- **⚡ Latency-Aware Pipeline** — Threaded audio with real-time latency profiling
- **🎛️ Multiple Waveforms** — Sine, sawtooth, square, triangle with band-limited anti-aliased synthesis
- **📊 Audio Visualizer** — Real-time waveform overlay

## 🎯 How It Works

```
[Webcam] → [MediaPipe Hands] → [Landmark Smoothing (EMA)]
                                       ↓
                              [Pinch Detection]  ←→  [ML Classifier]
                                       ↓
                              [Theremin Mapper]
                          X → Pitch (exp.)
                          Y → Volume (linear)
                          Spread → Filter cutoff
                                       ↓
                            [DSP Synthesizer]
                        Oscillator + ADSR + Filter
                                       ↓
                    [Audio Output + MIDI + Visualizer]
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Hand Tracking | MediaPipe Hands (21 landmarks, real-time) |
| Audio Synthesis | Pygame mixer + NumPy DSP (band-limited oscillators) |
| ML Gesture Recognition | sklearn MLP → ONNX Runtime inference |
| MIDI Output | mido (pitch_bend + CC to any DAW) |
| Smoothing | Exponential Moving Average (EMA) on landmarks |
| UI | Pygame overlay (frequency/volume bars, waveform visualizer) |
| Latency Profiling | Per-stage timing with avg/max/p95 reporting |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

**No webcam?** Run the headless smoke test:
```bash
python headless_smoke.py
```

## 🎮 Controls

| Action | Control |
|--------|---------|
| Control pitch | Move hand left/right (X-axis) |
| Control volume | Move hand up/down (Y-axis) |
| Start sound | Pinch thumb + index together |
| Stop sound | Release pinch |
| Bright timbre | Spread fingers open |
| Dark timbre | Close fingers / make fist |
| Cycle waveform | Press `W` |
| Quit | Press `ESC` |

## 🏗️ Architecture

```
ThereSyn/
├── main.py                    # Entry point, main loop
├── config.py                  # All configuration constants
│
├── vision/
│   ├── camera.py              # Webcam frame capture
│   └── hand_tracker.py        # MediaPipe hand landmark extraction
│
├── gesture/
│   ├── theremin_mapper.py     # Hand position → pitch/volume/filter mapping
│   ├── pinch_detector.py      # Pinch gesture detection (engage/disengage)
│   └── gesture_classifier.py  # ML-based gesture classifier (ONNX)
│
├── dsp/
│   ├── oscillator.py          # Band-limited waveform generation
│   ├── envelope.py            # ADSR envelope shaping
│   └── filter.py              # Lowpass filter for timbre control
│
├── engine/
│   ├── audio_engine.py        # Threaded audio synthesis pipeline
│   ├── midi_output.py         # MIDI stream (pitch_bend + CC)
│   └── latency_profiler.py    # Real-time latency measurement
│
├── ui/
│   ├── theremin_ui.py         # Theremin spatial visualization
│   └── visualizer.py          # Waveform display
│
├── utils/
│   ├── smoothing.py           # Landmark EMA filter
│   ├── debounce.py            # Trigger cooldown
│   └── logger.py              # Structured logging
│
├── tests/
│   └── ...                    # Unit tests
├── assets/
│   └── gesture_classifier.onnx  # Trained ML model
└── requirements.txt
```

## 📊 Latency Targets

| Pipeline Stage | Target |
|---------------|--------|
| Frame capture | < 5 ms |
| Hand detection | < 15 ms |
| Gesture classification | < 5 ms |
| Audio synthesis | < 10 ms |
| **End-to-end** | **< 50 ms** |

## 📖 Pipeline Detail

### Vision
- **Camera** (`vision/camera.py`) — OpenCV capture with disconnect tolerance (30-frame grace)
- **HandTracker** (`vision/hand_tracker.py`) — MediaPipe Hands, up to 2 hands, confidence-tunable
- **LandmarkSmoother** (`utils/smoothing.py`) — EMA filter (α=0.35) to reduce jitter

### Gesture
- **PinchDetector** (`gesture/pinch_detector.py`) — Euclidean distance between thumb tip (4) and index tip (8) → engage/disengage with edge detection
- **ThereminMapper** (`gesture/theremin_mapper.py`) — Exponential frequency mapping (C3–C6), linear volume, finger spread → filter cutoff
- **GestureClassifier** (`gesture/gesture_classifier.py`) — ONNX Runtime inference, 63-dim input (21 landmarks × 3 coords), 7 gesture classes, confidence thresholding

### DSP
- **Oscillator** (`dsp/oscillator.py`) — Band-limited saw/square/tri via additive synthesis, Nyquist-safe
- **Envelope** (`dsp/envelope.py`) — ADSR with configurable attack/decay/sustain/release
- **Filter** (`dsp/filter.py`) — State-variable lowpass for timbre control

### Audio Engine
- **AudioEngine** (`engine/audio_engine.py`) — Threaded synthesis loop, lock-free parameter updates, headless mode for testing
- **MIDIOutput** (`engine/midi_output.py`) — Real-time pitch_bend + CC7 (volume), configurable bend range
- **LatencyProfiler** (`engine/latency_profiler.py`) — Per-frame timing: vision → gesture → audio queue stages

### ML Training
- **train_model.py** — Three modes:
  - `--collect` — Interactive webcam sampler (key press to label, SPACE to record)
  - `--train` — sklearn MLPClassifier → ONNX export with embedded labels
  - `--verify` — Load ONNX model, run dummy inference, confirm output

## 📚 Related Work

- **Theremin** (Lev Termen, 1920) — The original contactless electronic instrument
- **MediaPipe Hands** (Google, 2020) — Real-time 21-point hand landmark detection
- **Hand Gesture Recognition survey** (arXiv:2408.05436) — Comprehensive review 2014–2024
- **Spatial audio in VR** (ACM, 2023) — Interaction paradigms for spatial music control
- **Human latency tolerance for gestural sound control** (Aalto, 2010) — Thresholds for acceptable gesture-to-audio delay

## 📄 License

MIT

---

## ✅ Real-hardware runbook

See REAL_HARDWARE_RUNBOOK.md for step-by-step verification on a machine with webcam, audio output, and optional MIDI routing (virtual loopback). It covers collecting samples, training the ONNX gesture model, verifying inference, and demo-recording tips using OBS/ffmpeg.

## 🎬 Recording a demo / GIF

1. Record 10–30s of the application using OBS (window or display capture). Capture system audio so the theremin sound is included.
2. Export a short MP4 (H.264). For GIFs, downsample to 15 fps and scale to ~800 px wide.
   Example:

   ffmpeg -i demo.mp4 -vf "fps=15,scale=800:-1:flags=lanczos" -y demo-15fps.gif
   gifsicle -O3 --colors 256 demo-15fps.gif -o demo-optimized.gif

3. Keep GIFs short (5–12s) for README use. Host large media externally if necessary.

---

Built by [Nirmit](https://github.com/nirmit7717)
