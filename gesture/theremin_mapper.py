"""
gesture/theremin_mapper.py — Maps hand spatial position to continuous theremin parameters.

Converts hand position in normalized camera coordinates to:
  - Pitch: X-axis position → frequency (exponential mapping for musical perception)
  - Volume: Y-axis position → amplitude (linear mapping)
  - Filter cutoff: Finger spread → timbre brightness
"""

import math
from config import (
    THEREMIN_FREQ_MIN, THEREMIN_FREQ_MAX,
    THEREMIN_VOL_MIN, THEREMIN_VOL_MAX,
    THEREMIN_X_DEADZONE, THEREMIN_Y_DEADZONE,
    THEREMIN_Y_INVERTED,
)


class ThereminMapper:
    """Maps hand position to continuous theremin control parameters."""

    def __init__(
        self,
        freq_min: float = THEREMIN_FREQ_MIN,
        freq_max: float = THEREMIN_FREQ_MAX,
        vol_min: float = THEREMIN_VOL_MIN,
        vol_max: float = THEREMIN_VOL_MAX,
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.vol_min = vol_min
        self.vol_max = vol_max

        # Precompute log range for exponential pitch mapping
        self._log_freq_min = math.log2(freq_min)
        self._log_freq_max = math.log2(freq_max)

        self._current_freq = freq_min
        self._current_vol = 0.0
        self._current_cutoff = 1.0
        self._smoothing_alpha = 0.35

    def map_hand(self, landmarks: list, engaged: bool) -> dict:
        """
        Map hand landmarks to theremin parameters.

        Args:
            landmarks: 21 smoothed (x, y, z) tuples.
            engaged: Whether the hand is currently pinch-engaged.

        Returns:
            {
                "pitch": float,        # Hz
                "volume": float,       # 0.0-1.0
                "engaged": bool,
                "filter_cutoff": float # 0.0-1.0 (timbre brightness)
            }
        """
        # Use wrist (landmark 0) as primary position reference
        wrist_x, wrist_y = landmarks[0][0], landmarks[0][1]

        # --- Pitch: exponential mapping ---
        norm_x = max(0.0, min(1.0, wrist_x))
        target_freq = 2 ** (self._log_freq_min + norm_x * (self._log_freq_max - self._log_freq_min))

        # --- Volume: linear mapping ---
        norm_y = max(0.0, min(1.0, wrist_y))
        if THEREMIN_Y_INVERTED:
            norm_y = 1.0 - norm_y
        target_vol = self.vol_min + norm_y * (self.vol_max - self.vol_min)

        # --- Filter cutoff: finger spread ---
        tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
        avg_dist = sum(
            math.dist((t[0], t[1]), (wrist_x, wrist_y)) for t in tips
        ) / len(tips)
        target_cutoff = max(0.0, min(1.0, (avg_dist - 0.08) / 0.25))

        # Smooth all parameters
        self._current_freq += self._smoothing_alpha * (target_freq - self._current_freq)
        if engaged:
            self._current_vol += self._smoothing_alpha * (target_vol - self._current_vol)
        else:
            self._current_vol += self._smoothing_alpha * (0.0 - self._current_vol)
        self._current_cutoff += self._smoothing_alpha * (target_cutoff - self._current_cutoff)

        return {
            "pitch": self._current_freq,
            "volume": self._current_vol if engaged else 0.0,
            "engaged": engaged,
            "filter_cutoff": self._current_cutoff,
        }

    def reset(self):
        """Reset to default state."""
        self._current_freq = self.freq_min
        self._current_vol = 0.0
        self._current_cutoff = 1.0
