# ThereSyn
### AR Theremin-Synthesizer — Play Music in Free Space

ThereSyn is a real-time gesture-based theremin that uses your hands to control pitch, volume, and timbre through a webcam — no physical contact required.

Built with MediaPipe, Pygame, NumPy, and real-time DSP.

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

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
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

## 📚 Related Work

- **Theremin** (Lev Termen, 1920) — The original contactless electronic instrument
- **MediaPipe Hands** (Google, 2020) — Real-time 21-point hand landmark detection
- **Hand Gesture Recognition survey** (arXiv:2408.05436) — Comprehensive review 2014–2024
- **Spatial audio in VR** (ACM, 2023) — Interaction paradigms for spatial music control
- **Human latency tolerance for gestural sound control** (Aalto, 2010) — Thresholds for acceptable gesture-to-audio delay

## 📄 License

MIT

---

Built by [Nirmit](https://github.com/nirmit7717)
