"""
main.py — ThereSyn entry point.

Orchestrates the full pipeline:
  Camera → HandTracker → LandmarkSmoother → PinchDetector → ThereminMapper
    → AudioEngine (continuous synthesis) + MIDIOutput (pitch_bend)
    → UI Overlay + LatencyProfiler + ML GestureClassifier
"""

import time
import numpy as np
import pygame

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    DEFAULT_WAVEFORM, LATENCY_LOG_ENABLED,
)

from vision.camera import Camera
from vision.hand_tracker import HandTracker
from utils.smoothing import LandmarkSmoother
from utils.debounce import Debouncer
from utils.logger import Logger

from gesture.theremin_mapper import ThereminMapper
from gesture.pinch_detector import PinchDetector
from gesture.gesture_classifier import GestureClassifier

from engine.audio_engine import AudioEngine
from engine.midi_output import MIDIOutput
from engine.latency_profiler import LatencyProfiler

from ui.theremin_ui import ThereminUI
from ui.visualizer import AudioVisualizer


def main():
    log = Logger("ThereSyn")

    # === Init pygame (centralized) ===
    pygame.init()
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(32)
    screen = pygame.display.set_mode((FRAME_WIDTH, FRAME_HEIGHT))
    pygame.display.set_caption("ThereSyn — AR Theremin-Synthesizer")
    clock = pygame.time.Clock()

    # === Vision ===
    try:
        camera = Camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    except Exception as e:
        log.error(f"Camera access failed: {e}")
        return

    hand_tracker = HandTracker()
    smoother = LandmarkSmoother()
    pinch_detector = PinchDetector()

    # === Gesture ===
    theremin_mapper = ThereminMapper()
    gesture_classifier = GestureClassifier()

    # === Engine ===
    audio_engine = AudioEngine()
    midi_output = MIDIOutput()
    latency_profiler = LatencyProfiler(report_interval=LATENCY_LOG_INTERVAL) if LATENCY_LOG_ENABLED else None

    # === UI ===
    theremin_ui = ThereminUI(FRAME_WIDTH, FRAME_HEIGHT)
    visualizer = AudioVisualizer(FRAME_WIDTH, 60)

    # === State ===
    current_waveform = DEFAULT_WAVEFORM
    waveform_cycle = ["sine", "sawtooth", "square", "triangle"]
    waveform_idx = 0
    system_debouncer = Debouncer(debounce_time=0.5)

    # ML-driven octave offset (in semitones)
    octave_offset = 0

    log.info("ThereSyn started. Press ESC to quit.")
    log.info("Controls: Pinch to engage | Move hand for pitch/vol | W = cycle waveform | ESC = quit")

    running = True

    while running:
        # --- Profiler: frame start ---
        if latency_profiler:
            latency_profiler.start_frame()

        frame = camera.read_frame()
        if frame is None:
            log.warn("Failed to read frame")
            break

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w and system_debouncer.can_trigger("waveform"):
                    waveform_idx = (waveform_idx + 1) % len(waveform_cycle)
                    current_waveform = waveform_cycle[waveform_idx]
                    audio_engine.set_waveform(current_waveform)
                    log.info(f"Waveform: {current_waveform}")

        # --- Vision pipeline ---
        hands_data = hand_tracker.process_frame(frame)

        # Draw landmarks on raw frame (before smoothing) for visual feedback
        for hand in hands_data:
            if "raw_landmarks" in hand:
                import mediapipe as mp
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand["raw_landmarks"],
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        smoothed = smoother.smooth(hands_data)
        pinch_state = pinch_detector.detect(smoothed)

        # --- Profiler: vision done ---
        if latency_profiler:
            latency_profiler.mark("vision")

        # --- ML gesture classification (optional: auto-engage features) ---
        ml_result = None
        if gesture_classifier and smoothed:
            ml_result = gesture_classifier.classify(smoothed)
            if ml_result:
                g = ml_result.get("gesture")
                conf = ml_result.get("confidence", 0.0)
                # ML override: force disengage
                if g == "stop" and conf >= gesture_classifier.confidence_threshold:
                    for pid in pinch_state:
                        pinch_state[pid]["pinching"] = False
                # ML auto-engage (if configured and confident)
                elif g == "theremin_engage" and conf >= gesture_classifier.confidence_threshold:
                    for pid in pinch_state:
                        pinch_state[pid]["pinching"] = True
                # ML octave controls (debounced)
                elif g in ("octave_up", "octave_down") and conf >= gesture_classifier.confidence_threshold:
                    if system_debouncer.can_trigger(g):
                        if g == "octave_up":
                            octave_offset += 12
                            log.info("ML: octave up")
                        else:
                            octave_offset -= 12
                            log.info("ML: octave down")

        # --- Theremin processing ---
        theremin_data = None
        pinch_pos = None

        if smoothed:
            # Use primary hand (first detected)
            hand_landmarks = smoothed[0]["landmarks"]
            engaged = pinch_state.get(0, {}).get("pinching", False)

            theremin_data = theremin_mapper.map_hand(hand_landmarks, engaged)

            # Get pinch position in pixel coords for UI
            if 0 in pinch_state:
                cx, cy = pinch_state[0]["center"]
                pinch_pos = (int(cx * FRAME_WIDTH), int(cy * FRAME_HEIGHT))

            # Apply octave offset (from ML) to pitch
            adj_pitch = theremin_data["pitch"] * (2 ** (octave_offset / 12.0))

            # Send to audio + MIDI
            audio_engine.update_theremin(
                frequency=adj_pitch,
                volume=theremin_data["volume"],
                filter_cutoff=theremin_data["filter_cutoff"],
                waveform=current_waveform,
            )

            if engaged:
                audio_engine.theremin_engage()
                if midi_output and midi_output.enabled:
                    midi_note = midi_output.freq_to_midi(adj_pitch)
                    bend = midi_output.freq_to_pitch_bend(adj_pitch, midi_note)
                    midi_output.pitch_bend(bend)
                    midi_output.control_change(7, int(theremin_data["volume"] * 127))
            else:
                audio_engine.theremin_disengage()
                if midi_output and midi_output.enabled:
                    midi_output.pitch_bend(8192)

        # --- Profiler: gesture + audio done ---
        if latency_profiler:
            latency_profiler.mark("audio_queue")
            report = latency_profiler.end_frame()
            if report:
                log.latency_report(report)

        # --- Render ---
        frame_rgb = frame[:, :, ::-1]  # BGR → RGB
        frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
        frame_surf = pygame.surfarray.make_surface(frame_transposed)

        # Theremin UI overlay
        if theremin_data:
            theremin_ui.draw(frame_surf, theremin_data, pinch_pos)

        # Waveform visualizer
        visualizer.draw_waveform(frame_surf, x=0, y=0)

        # Screen
        screen.blit(frame_surf, (0, 0))
        pygame.display.flip()

        clock.tick(TARGET_FPS)

    # --- Cleanup ---
    camera.release()
    audio_engine.quit()
    if midi_output:
        midi_output.close()
    pygame.quit()
    log.info("ThereSyn stopped.")


if __name__ == "__main__":
    main()
