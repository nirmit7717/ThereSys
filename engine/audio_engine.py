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
from dsp.oscillator import Oscillator
from dsp.envelope import ADSREnvelope
from dsp.filter import LowpassFilter
from config import (
    SAMPLE_RATE, BUFFER_SIZE, CHANNELS,
    DEFAULT_WAVEFORM,
)


class AudioEngine:
    """Threaded audio engine for continuous theremin synthesis.

    Supports dependency injection for headless testing via optional parameters:
      oscillator, envelope, filter, enable_audio
    """

    def __init__(self, oscillator=None, envelope=None, filter=None, enable_audio: bool = True):
        self._enable_audio = enable_audio

        # Initialize pygame mixer only if audio enabled and mixer not already initialized by main
        if self._enable_audio:
            try:
                import pygame as _pygame
                if not _pygame.mixer.get_init():
                    _pygame.mixer.init()
                    _pygame.mixer.set_num_channels(32)
                self._pygame = _pygame
            except Exception as e:
                print(f"[Audio] Warning: mixer init failed or pygame unavailable: {e}")
                self._enable_audio = False
                self._pygame = None
        else:
            self._pygame = None

        # Allow injected DSP components for tests to avoid heavy deps
        if oscillator is not None:
            self.oscillator = oscillator
        elif self._enable_audio:
            self.oscillator = Oscillator(SAMPLE_RATE)
        else:
            self.oscillator = None

        if envelope is not None:
            self.envelope = envelope
        elif self._enable_audio:
            self.envelope = ADSREnvelope(attack=0.05, decay=0.1, sustain_level=0.8, release=0.3)
        else:
            self.envelope = None

        if filter is not None:
            self.filter = filter
        elif self._enable_audio:
            self.filter = LowpassFilter(SAMPLE_RATE)
        else:
            self.filter = None

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

            # If audio playback is disabled (headless tests), skip synthesis but keep state updated
            if self._enable_audio and self._engaged and self._vol > 0.01 and self.oscillator is not None:
                try:
                    import numpy as _np
                    sound_data = self.oscillator.generate(self._waveform, self._freq, 0.1, self._vol)
                    shaped = self.envelope.apply(sound_data)
                    shaped = self.filter.apply(shaped, self._filter_cutoff)
                    stereo = _np.column_stack((shaped, shaped))
                    int16 = (stereo * 32767).astype(_np.int16)
                    sound = self._pygame.sndarray.make_sound(int16)
                    sound.play()
                except Exception as e:
                    # Log and continue; playback failure shouldn't crash the audio thread
                    print(f"[Audio] Playback error: {e}")

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
        """Signal audio thread to stop and wait briefly. Do not call pygame.quit() here — main() owns pygame lifecycle."""
        self._queue.put(None)
        self._thread.join(timeout=1.0)
