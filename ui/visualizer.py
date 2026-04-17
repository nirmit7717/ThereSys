"""
ui/visualizer.py — Real-time audio waveform and spectrum display.

TODO (Collaborator B): Implement with actual audio data capture.
"""

import numpy as np
import pygame


class AudioVisualizer:
    def __init__(self, width: int = 640, height: int = 80):
        self.width = width
        self.height = height
        self._waveform = np.zeros(width)
        self._spectrum = np.zeros(width // 2)

    def update_waveform(self, data: np.ndarray | None):
        """Update waveform display data (mono float64 array)."""
        if data is not None and len(data) > 0:
            # Downsample to display width
            indices = np.linspace(0, len(data) - 1, self.width, dtype=int)
            self._waveform = data[indices]
        else:
            self._waveform *= 0.9  # decay when no data

    def update_spectrum(self, data: np.ndarray | None):
        """Update frequency spectrum from audio samples."""
        if data is not None and len(data) > 64:
            spectrum = np.abs(np.fft.rfft(data[:2048]))
            spectrum = spectrum[:self.width // 2]
            peak = np.max(spectrum)
            if peak > 0:
                spectrum = spectrum / peak
            # Smooth
            self._spectrum = self._spectrum * 0.7 + spectrum * 0.3
        else:
            self._spectrum *= 0.9

    def draw_waveform(self, surface, x: int = 0, y: int = 0):
        """Draw waveform overlay."""
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 80))
        surface.blit(panel, (x, y))

        # Draw waveform line
        center_y = self.height // 2
        points = []
        for i in range(self.width):
            px = x + i
            py = y + center_y - int(self._waveform[i] * (self.height // 2 - 4))
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(surface, (0, 255, 100), False, points, 2)

    def draw_spectrum(self, surface, x: int = 0, y: int = 0):
        """Draw frequency spectrum bars."""
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 80))
        surface.blit(panel, (x, y))

        n_bars = len(self._spectrum)
        bar_w = max(1, self.width // n_bars)
        for i, magnitude in enumerate(self._spectrum):
            bx = x + i * bar_w
            h = int(magnitude * (self.height - 4))
            if h > 0:
                color = (
                    min(255, int(100 + 155 * (i / n_bars))),
                    max(0, int(255 - 200 * (i / n_bars))),
                    100,
                )
                pygame.draw.rect(surface, color, (bx, y + self.height - h - 2, bar_w - 1, h))
