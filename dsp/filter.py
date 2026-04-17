"""
dsp/filter.py — Simple digital filters for real-time audio processing.

Used in theremin mode for timbre control (finger spread → filter cutoff).
"""

import numpy as np
from config import SAMPLE_RATE


class LowpassFilter:
    """Simple one-pole lowpass filter for real-time timbre control."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._prev = 0.0

    def apply(self, wave: np.ndarray, cutoff_normalized: float) -> np.ndarray:
        """
        Apply lowpass filter to waveform.

        Args:
            wave: Mono float64 array.
            cutoff_normalized: 0.0 (fully open) to 1.0 (heavily filtered).
                              Maps to actual cutoff frequency internally.

        Returns:
            Filtered waveform.
        """
        # Map normalized cutoff to frequency range: 200Hz - 20000Hz
        cutoff_hz = 200 * (1.0 - cutoff_normalized) + 20000 * cutoff_normalized
        # Actually invert: cutoff_normalized=0 → dark (low cutoff), 1 → bright (high cutoff)
        cutoff_hz = 200 + cutoff_normalized * 19800

        # One-pole IIR filter coefficient
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)

        output = np.zeros_like(wave)
        self._prev = 0.0

        for i in range(len(wave)):
            self._prev = self._prev + alpha * (wave[i] - self._prev)
            output[i] = self._prev

        return output
