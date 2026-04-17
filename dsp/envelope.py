"""
dsp/envelope.py — ADSR envelope generator for discrete notes.

Provides Attack-Decay-Sustain-Release shaping for piano mode notes.
"""

import numpy as np
from config import SAMPLE_RATE


class ADSREnvelope:
    """Generates ADSR envelopes for waveform shaping."""

    def __init__(
        self,
        attack: float = 0.01,
        decay: float = 0.15,
        sustain_level: float = 0.6,
        release: float = 0.2,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        self.sample_rate = sample_rate

    def apply(self, wave: np.ndarray) -> np.ndarray:
        """
        Apply ADSR envelope to a waveform.

        Args:
            wave: Mono float64 numpy array.

        Returns:
            Envelope-shaped wave (same shape).
        """
        n = len(wave)
        envelope = np.ones(n, dtype=np.float64)

        a = int(self.attack * self.sample_rate)
        d = int(self.decay * self.sample_rate)
        r = int(self.release * self.sample_rate)

        idx = 0

        # Attack
        if a > 0:
            end = min(a, n)
            envelope[idx:end] = np.linspace(0.0, 1.0, end - idx)
            idx = end

        # Decay
        if d > 0 and idx < n:
            end = min(idx + d, n)
            envelope[idx:end] = np.linspace(1.0, self.sustain_level, end - idx)
            idx = end

        # Sustain (fill until release zone)
        if idx < n - r:
            envelope[idx:n - r] = self.sustain_level
            idx = n - r

        # Release
        if r > 0 and idx < n:
            end = n
            start_val = self.sustain_level
            envelope[idx:end] = np.linspace(start_val, 0.0, end - idx)

        return wave * envelope

    def apply_with_release(self, wave: np.ndarray, release_early: bool = False) -> np.ndarray:
        """
        Apply ADSR with optional early release (note cut short).

        When release_early=True, skip sustain and go straight to release from current level.
        """
        if not release_early:
            return self.apply(wave)

        n = len(wave)
        envelope = np.ones(n, dtype=np.float64)

        a = int(self.attack * self.sample_rate)
        d = int(self.decay * self.sample_rate)
        r = int(self.release * self.sample_rate)

        idx = 0

        # Attack
        if a > 0:
            end = min(a, n)
            envelope[idx:end] = np.linspace(0.0, 1.0, end - idx)
            idx = end

        # Decay → immediately into release
        if d > 0 and idx < n:
            end = min(idx + d, n)
            decay_env = np.linspace(1.0, 0.0, end - idx)
            envelope[idx:end] = decay_env
            idx = end

        # Zero the rest
        if idx < n:
            envelope[idx:] = 0.0

        return wave * envelope
