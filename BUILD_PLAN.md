# ThereSyn — Build Plan (2 People, ~2 Days)

## Timeline: 2 days (~16-20 hours total)

---

## 🏗️ Architecture

```
Camera → HandTracker → EMA Smoother → PinchDetector
                                        ↓
                              ThereminMapper
                          ┌────────────────────┐
                          │ X → pitch (exp.)    │
                          │ Y → volume (linear) │
                          │ spread → filter     │
                          └─────────┬──────────┘
                                    ↓
                           AudioEngine (threaded)
                        Oscillator + ADSR + Filter
                                    ↓
                      ┌─────────────┼─────────────┐
                      ▼             ▼             ▼
                 AudioOut      MIDIOutput    LatencyProfiler
                      │
                      ▼
              UI (ThereminUI + Visualizer + Landmarks)
```

### Key Design Decisions

1. **Exponential pitch mapping** — X position maps to frequency via `2^x`, not linearly. This matches how humans perceive pitch (octaves are exponential). A linear slider would compress bass notes and stretch treble.

2. **Pinch-to-engage, not hover** — Sound only plays while pinching. Moving your hand without pinching does nothing. This prevents accidental sound and gives explicit control, matching how a real theremin has a volume antenna you must physically approach.

3. **Band-limited synthesis** — Sawtooth/square waves are generated via additive harmonics with a Nyquist cap. Naive `sign(sin(x))` sawtooth aliases badly at high frequencies and sounds harsh.

4. **Threaded audio** — Audio synthesis runs on a dedicated thread via command queue. The main loop never blocks on `pygame.sndarray.make_sound()`.

5. **EMA smoothing on landmarks** — All 63 landmark coordinates (21 points × 3 axes) pass through independent exponential moving average filters before any gesture logic. Eliminates jitter from MediaPipe without adding perceptible latency.

6. **MIDI pitch_bend for microtonal slides** — Theremin produces continuous frequencies between MIDI notes. We send MIDI `note_on` at the nearest integer note + `pitch_bend` for the fractional part. This is how professional MIDI theremins work.

---

## 📋 Phase Breakdown

### Phase 1: Core Pipeline — Get Sound Playing (3-4 hours)
**Goal:** Camera → hand tracking → pinch → pitch control → audio output. MVP.

| Task | Owner | Files | Est. |
|------|-------|-------|------|
| 1.1 Camera + HandTracker | Nirmit | `vision/camera.py`, `vision/hand_tracker.py` | 30m |
| 1.2 Landmark EMA smoothing | Collaborator | `utils/smoothing.py` | 30m |
| 1.3 Pinch detector | Collaborator | `gesture/pinch_detector.py` | 30m |
| 1.4 Theremin mapper (X→pitch, Y→vol) | Collaborator | `gesture/theremin_mapper.py` | 1h |
| 1.5 Audio engine (threaded, continuous) | Nirmit | `engine/audio_engine.py` | 1h |
| 1.6 DSP oscillator (sine first) | Collaborator | `dsp/oscillator.py` | 45m |
| 1.7 ADSR envelope | Collaborator | `dsp/envelope.py` | 30m |
| 1.8 Main loop wiring | Nirmit | `main.py` | 45m |

**Checkpoint:** Hand in camera → pinch to engage → move hand → hear pitch change. No UI yet, just terminal logs.

**Dependencies:**
- 1.1 → 1.2 → 1.3 → 1.4 (collaborator chain)
- 1.5 depends on 1.6 + 1.7 (oscillator + envelope)
- 1.8 depends on everything above

---

### Phase 2: UI + Polish + Filters (4-5 hours)
**Goal:** Visual feedback, waveform selection, timbre control, proper UX.

| Task | Owner | Files | Est. |
|------|-------|-------|------|
| 2.1 Theremin UI (freq display, vol bar, pinch indicator) | Collaborator | `ui/theremin_ui.py` | 1.5h |
| 2.2 Landmark overlay on frame | Nirmit | `main.py` | 20m |
| 2.3 Lowpass filter (finger spread → timbre) | Collaborator | `dsp/filter.py` | 45m |
| 2.4 Additional waveforms (saw, square, triangle) | Collaborator | `dsp/oscillator.py` (extend) | 45m |
| 2.5 Waveform cycling (W key + UI indicator) | Nirmit | `main.py` | 20m |
| 2.6 Audio visualizer (waveform overlay) | Collaborator | `ui/visualizer.py` | 1h |

**Checkpoint:** Full theremin experience with visual feedback. Looks good in a demo.

---

### Phase 3: ML + MIDI + Latency (4-5 hours)
**Goal:** Intelligence layer, DAW integration, performance measurement.

| Task | Owner | Files | Est. |
|------|-------|-------|------|
| 3.1 MIDI output (pitch_bend + CC) | Collaborator | `engine/midi_output.py` | 1h |
| 3.2 Wire MIDI into main loop | Nirmit | `main.py` | 30m |
| 3.3 Latency profiler + reporting | Nirmit | `engine/latency_profiler.py` | 1h |
| 3.4 ML data collector (record gestures) | Collaborator | `gesture/gesture_classifier.py` | 1h |
| 3.5 Train classifier (sklearn MLP → ONNX) | Collaborator | `gesture/gesture_classifier.py` | 1.5h |
| 3.6 Wire ML classifier into main loop | Nirmit | `main.py` | 30m |

**Checkpoint:** MIDI streams to DAW. Latency numbers are reported. ML model classifies gestures.

---

### Phase 4: Integration + Demo (2-3 hours)
**Goal:** Bug fixes, edge cases, demo video, README final.

| Task | Owner | Est. |
|------|-------|------|
| 4.1 End-to-end testing + bug fixes | Nirmit | 1h |
| 4.2 Unit tests (smoothing, debounce, oscillator, freq calc) | Collaborator | 1h |
| 4.3 Demo video recording | Nirmit | 30m |
| 4.4 README + GIF + final polish | Both | 30m |

---

## 👥 Work Split

```
            NIRMIT
            ═════
  Phase 1: Camera, HandTracker, AudioEngine (threaded), Main loop wiring
  Phase 2: Landmark overlay, Waveform cycling, main.py integrations
  Phase 3: MIDI wiring, Latency profiler, ML wiring
  Phase 4: Integration testing, bug fixes, demo video

            COLLABORATOR
            ═══════════
  Phase 1: Smoothing, Pinch detector, Theremin mapper, DSP (oscillator + ADSR)
  Phase 2: Theremin UI, Filter, Additional waveforms, Visualizer
  Phase 3: MIDI output, ML data collector + classifier
  Phase 4: Unit tests, README polish
```

### Critical Path

```
Collaborator: Smoothing → Pinch → ThereminMapper → (done with Phase 1 core)
             Oscillator → ADSR → AudioEngine ← Nirmit waits here
Nirmit: Camera → (waits for Collab DSP) → AudioEngine → Main loop
```

**Parallel start:** Both begin immediately. Nirmit does camera + tracker (30m). Collaborator does smoothing + pinch (1h). They converge at AudioEngine which Nirmit owns but depends on Collaborator's DSP modules.

**Handoff points:**
1. After Phase 1 — test the full pipeline together
2. After Phase 2 — demo the visual experience
3. After Phase 3 — test MIDI + ML end-to-end

---

## 🔑 Interface Contracts (agree BEFORE splitting)

### ThereminMapper output (gesture → engine)
```python
{
    "pitch": 440.0,            # Hz (continuous float)
    "volume": 0.7,             # 0.0–1.0
    "engaged": True,           # pinch state
    "filter_cutoff": 0.8       # 0.0–1.0 (finger spread → timbre)
}
```

### AudioEngine API (thread-safe)
```python
audio_engine.update_theremin(frequency, volume, filter_cutoff, waveform)
audio_engine.theremin_engage()
audio_engine.theremin_disengage()
audio_engine.set_waveform("sawtooth")
```

### MIDIOutput API
```python
midi.pitch_bend(value)        # 0–16383, 8192 = center
midi.control_change(cc, value) # CC#7 = volume, CC#74 = filter cutoff
midi.note_on(note, velocity)
midi.note_off(note)
```

### PinchDetector output
```python
{
    0: {  # hand index
        "pinching": bool,
        "just_pinched": bool,
        "just_released": bool,
        "center": (x, y),      # normalized coords
        "distance": float
    }
}
```

---

## 🎯 What Makes This Resume-Worthy

| Aspect | Details |
|--------|---------|
| **Novelty** | Computer vision theremin with ML gesture classification + MIDI output. No existing open-source project does all three. |
| **DSP depth** | Band-limited synthesis, exponential pitch mapping, ADSR, real-time filtering — not just `pygame.mixer` playback |
| **ML engineering** | Data collection → training → ONNX export → inference pipeline |
| **Systems depth** | Threaded audio, latency profiling with p95 reporting |
| **Standards compliance** | MIDI pitch_bend for microtonal control, ONNX model format |
| **Measurable** | End-to-end latency benchmarks, frequency accuracy |
