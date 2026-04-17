"""
engine/audio_engine.py — Threaded audio output pipeline for ThereSyn.

Focused on continuous theremin synthesis:
  - Real-time waveform generation on demand
  - Smooth frequency/volume transitions
  - ADSR for engage/disengage
  - Lowpass filter for timbre control
"""

import threading
import queue
import numpy as np
import pygame
from dsp.oscillator import Oscillator
from dsp.envelope import ADSREnvelope
from dsp.filter import LowpassFilter
from config import (
    SAMPLE_RATE, BUFFER_SIZE, CHANNELS,
    DEFAULT_WAVEFORM,
)


class AudioEngine:
    """Threaded audio engine for continuous theremin synthesis."""

    def __init__(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, CHANNELS, BUFFER_SIZE)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)

        self.oscillator = Oscillator(SAMPLE_RATE)
        self.envelope = ADSREnvelope(attack=0.05, decay=0.1, sustain_level=0.8, release=0.3)
        self.filter = LowpassFilter(SAMPLE_RATE)

        # Theremin continuous state
        self._engaged = False
        self._freq = 440.0
        self._vol = 0.0
        self._waveform = DEFAULT_WAVEFORM
        self._filter_cutoff = 1.0

        # Audio command queue
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._audio_loop, daemon=True, name="AudioWorker")
        self._thread.start()

    def _audio_loop(self):
        """Background thread processing audio commands."""
        while True:
            cmd = self._queue.get()
            if cmd is None:
                break
            self._process(cmd)

    def _process(self, cmd: dict):
        action = cmd["action"]

        if action == "theremin_update":
            self._freq = cmd["frequency"]
            self._vol = cmd["volume"]
            self._filter_cutoff = cmd.get("filter_cutoff", 1.0)
            if "waveform" in cmd:
                self._waveform = cmd["waveform"]

            # Synthesize a short chunk and play
            if self._engaged and self._vol > 0.01:
                chunk_duration = 0.1  # 100ms chunks for smooth updates
                raw = self.oscillator.generate(self._waveform, self._freq, chunk_duration, self._vol)
                shaped = self.envelope.apply(raw)
                # Apply filter
                shaped = self.filter.apply(shaped, self._filter_cutoff)
                stereo = np.column_stack((shaped, shaped))
                int16 = (stereo * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(int16)
                sound.play()

        elif action == "theremin_engage":
            self._engaged = True

        elif action == "theremin_disengage":
            self._engaged = False

        elif action == "set_waveform":
            self._waveform = cmd["waveform"]

    # === Public API (thread-safe, queued) ===

    def update_theremin(self, frequency: float, volume: float,
                        filter_cutoff: float = 1.0, waveform: str = None):
        self._queue.put({
            "action": "theremin_update",
            "frequency": frequency,
            "volume": volume,
            "filter_cutoff": filter_cutoff,
            "waveform": waveform,
        })

    def theremin_engage(self):
        self._queue.put({"action": "theremin_engage"})

    def theremin_disengage(self):
        self._queue.put({"action": "theremin_disengage"})

    def set_waveform(self, waveform: str):
        self._queue.put({"action": "set_waveform", "waveform": waveform})

    def quit(self):
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        pygame.quit()
