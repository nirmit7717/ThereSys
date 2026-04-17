"""
engine/latency_profiler.py — Real-time pipeline latency measurement.

Timestamps each stage of the processing pipeline and reports
average/max latency periodically.
"""

import time
import statistics


class LatencyProfiler:
    """Measures end-to-end and per-stage latency."""

    def __init__(self, report_interval: float = 2.0):
        self.report_interval = report_interval
        self.enabled = True

        # Per-frame timestamps (set and read within same frame)
        self._frame_start = 0.0
        self._stages = {}

        # Accumulated measurements
        self._measurements = {
            "frame_total": [],
            "vision": [],
            "gesture": [],
            "audio_queue": [],
        }

        self._last_report = time.time()

    def start_frame(self):
        """Call at the start of each frame."""
        self._frame_start = time.perf_counter()
        self._stages = {}

    def mark(self, stage: str):
        """Mark the end of a pipeline stage."""
        self._stages[stage] = time.perf_counter()

    def end_frame(self):
        """Call at the end of each frame. Computes stage latencies."""
        now = time.perf_counter()
        frame_total = now - self._frame_start

        self._measurements["frame_total"].append(frame_total * 1000)  # ms

        if "vision" in self._stages:
            self._measurements["vision"].append(
                (self._stages["vision"] - self._frame_start) * 1000
            )

        prev = self._frame_start
        for stage in ["gesture", "audio_queue"]:
            if stage in self._stages:
                self._measurements[stage].append(
                    (self._stages[stage] - prev) * 1000
                )
                prev = self._stages[stage]

        # Prune old measurements (keep last 300 frames)
        for key in self._measurements:
            if len(self._measurements[key]) > 300:
                self._measurements[key] = self._measurements[key][-300:]

        # Periodic report
        if now - self._last_report >= self.report_interval:
            self._last_report = now
            return self.get_report()
        return None

    def get_report(self) -> dict | None:
        """Get current latency statistics."""
        if not self.enabled:
            return None

        report = {}
        for key, values in self._measurements.items():
            if not values:
                report[key] = None
                continue
            report[key] = {
                "avg_ms": round(statistics.mean(values), 1),
                "max_ms": round(max(values), 1),
                "min_ms": round(min(values), 1),
                "p95_ms": round(sorted(values)[int(len(values) * 0.95)], 1),
            }
        return report

    def get_last_frame_ms(self) -> float:
        """Get the most recent total frame time in ms."""
        if self._measurements["frame_total"]:
            return self._measurements["frame_total"][-1]
        return 0.0
