"""
utils/debounce.py — Trigger cooldown utility.

Fixed from old project: properly tracks per-key cooldown.
"""

import time
from collections import defaultdict


class Debouncer:
    """Prevents rapid re-triggering of the same event."""

    def __init__(self, debounce_time: float = 0.150):
        self.debounce_time = debounce_time
        self._last_triggered: dict[str, float] = {}

    def can_trigger(self, key: str) -> bool:
        """
        Check if enough time has passed since last trigger for this key.
        Updates the timestamp if trigger is allowed.
        """
        now = time.time()
        last = self._last_triggered.get(key, 0)
        if now - last >= self.debounce_time:
            self._last_triggered[key] = now
            return True
        return False

    def reset(self, key: str = None):
        """Reset cooldown for a specific key or all keys."""
        if key:
            self._last_triggered.pop(key, None)
        else:
            self._last_triggered.clear()
