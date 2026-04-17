"""
utils/smoothing.py — Exponential Moving Average filter for hand landmarks.

Eliminates jitter from MediaPipe landmark output without adding significant latency.
Each landmark coordinate is independently smoothed using EMA.
"""

from config import LANDMARK_SMOOTH_ALPHA


class LandmarkSmoother:
    """Applies EMA smoothing to hand landmark positions."""

    def __init__(self, alpha: float = LANDMARK_SMOOTH_ALPHA):
        """
        Args:
            alpha: Smoothing factor. 0.0 = no update (max smoothing), 1.0 = no smoothing.
        """
        self.alpha = alpha
        # Store smoothed state per hand: { hand_index: [(x,y,z), ...] }
        self._state = {}

    def reset(self):
        """Clear all smoothed state (call on hand loss/re-acquire)."""
        self._state.clear()

    def smooth(self, hands_data: list, track_ids: list | None = None) -> list:
        """
        Apply EMA smoothing to landmark positions.

        Args:
            hands_data: List of hand dicts from HandTracker.
                        Each has "type", "landmarks", "raw_landmarks".
            track_ids: Optional list of hand IDs for stable tracking across frames.

        Returns:
            New list with smoothed landmarks (same dict structure, landmarks replaced).
            "raw_landmarks" is preserved for UI drawing.
        """
        result = []
        for i, hand in enumerate(hands_data):
            key = track_ids[i] if track_ids else i
            landmarks = hand["landmarks"]

            if key not in self._state or len(self._state[key]) != len(landmarks):
                # First frame for this hand — initialize directly
                self._state[key] = [(x, y, z) for x, y, z in landmarks]
            else:
                # Apply EMA
                smoothed = []
                for j, (x, y, z) in enumerate(landmarks):
                    sx, sy, sz = self._state[key][j]
                    nx = sx + self.alpha * (x - sx)
                    ny = sy + self.alpha * (y - sy)
                    nz = sz + self.alpha * (z - sz)
                    smoothed.append((nx, ny, nz))
                self._state[key] = smoothed

            result.append({
                "type": hand["type"],
                "landmarks": self._state[key],
                "raw_landmarks": hand.get("raw_landmarks"),
            })

        # Prune stale hand entries
        active_keys = set(track_ids if track_ids else range(len(hands_data)))
        stale = [k for k in self._state if k not in active_keys]
        for k in stale:
            del self._state[k]

        return result
