"""
gesture/pinch_detector.py — Detect pinch gestures (thumb tip + index tip proximity).

Used for:
  - Theremin engage/disengage
  - System button activation (octave, instrument)
  - Piano drag handle
"""

import math
from config import PINCH_THRESHOLD


class PinchDetector:
    """Detects pinch gesture between thumb tip (landmark 4) and index tip (landmark 8)."""

    def __init__(self, threshold: float = PINCH_THRESHOLD):
        self.threshold = threshold
        # Track pinch state per hand for edge detection (was_pinched → not, etc.)
        self._prev_pinched = {}

    def detect(self, hands_data: list) -> dict:
        """
        Detect pinch state for each hand.

        Args:
            hands_data: List of hand dicts with "landmarks" (smoothed).

        Returns:
            Dict mapping hand index → {
                "pinching": bool,           # currently pinching?
                "just_pinched": bool,       # pinch started this frame?
                "just_released": bool,      # pinch released this frame?
                "center": (x, y),           # pinch midpoint in normalized coords
                "distance": float           # thumb-index distance
            }
        """
        results = {}

        for i, hand in enumerate(hands_data):
            landmarks = hand["landmarks"]
            thumb_tip = landmarks[4]   # (x, y, z)
            index_tip = landmarks[8]

            dist = math.dist(
                (thumb_tip[0], thumb_tip[1]),
                (index_tip[0], index_tip[1])
            )
            is_pinching = dist < self.threshold

            was_pinching = self._prev_pinched.get(i, False)

            results[i] = {
                "pinching": is_pinching,
                "just_pinched": is_pinching and not was_pinching,
                "just_released": not is_pinching and was_pinching,
                "center": (
                    (thumb_tip[0] + index_tip[0]) / 2,
                    (thumb_tip[1] + index_tip[1]) / 2,
                ),
                "distance": dist,
            }

            self._prev_pinched[i] = is_pinching

        return results
