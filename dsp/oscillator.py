"""
dsp/oscillator.py — Real-time waveform generation.

Supports sine, sawtooth, square, and triangle waves with anti-aliasing
(band-limited synthesis via additive harmonics).
"""

import numpy as np
from config import SAMPLE_RATE


class Oscillator:
    """Generates band-limited waveforms at arbitrary frequencies."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        # Max harmonics before aliasing at 20kHz
        self._max_harmonics = int(sample_rate / 2 / 20)  # ~1100 for 44100Hz

    def generate(self, waveform: str, frequency: float, duration: float, volume: float = 0.5) -> np.ndarray:
        """
        Generate a waveform array.

        Args:
            waveform: "sine" | "sawtooth" | "square" | "triangle"
            frequency: Pitch in Hz.
            duration: Length in seconds.
            volume: Amplitude 0.0-1.0.

        Returns:
            Mono float64 numpy array, normalized to [-1, 1] * volume.
        """
        n = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)

        generators = {
            "sine": self._sine,
            "sawtooth": self._sawtooth,
            "square": self._square,
            "triangle": self._triangle,
        }

        gen = generators.get(waveform, self._sine)
        wave = gen(t, frequency)

        # Normalize
        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = wave / peak

        return wave * volume

    def _sine(self, t, freq):
        return np.sin(2 * np.pi * freq * t)

    def _sawtooth(self, t, freq):
        """Band-limited sawtooth via additive synthesis."""
        wave = np.zeros_like(t)
        n_harmonics = min(int(self.sample_rate / (2 * freq)), self._max_harmonics)
        for k in range(1, n_harmonics + 1):
            wave += ((-1) ** (k + 1)) * np.sin(2 * np.pi * freq * k * t) / k
        return wave

    def _square(self, t, freq):
        """Band-limited square via additive synthesis (odd harmonics)."""
        wave = np.zeros_like(t)
        n_harmonics = min(int(self.sample_rate / (2 * freq)), self._max_harmonics)
        for k in range(1, n_harmonics + 1, 2):
            wave += np.sin(2 * np.pi * freq * k * t) / k
        return wave

    def _triangle(self, t, freq):
        """Band-limited triangle via additive synthesis (odd harmonics, 1/n²)."""
        wave = np.zeros_like(t)
        n_harmonics = min(int(self.sample_rate / (2 * freq)), self._max_harmonics)
        for k in range(1, n_harmonics + 1, 2):
            sign = 1 if ((k - 1) // 2) % 2 == 0 else -1
            wave += sign * np.sin(2 * np.pi * freq * k * t) / (k * k)
        return wave

    def generate_for_mixer(self, waveform: str, frequency: float, duration: float, volume: float = 0.5) -> np.ndarray:
        """
        Generate stereo int16 array ready for pygame.sndarray.make_sound().
        """
        mono = self.generate(waveform, frequency, duration, volume)
        stereo = np.column_stack((mono, mono))
        int16 = (stereo * 32767).astype(np.int16)
        return int16
