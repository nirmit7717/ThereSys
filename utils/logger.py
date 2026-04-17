"""
utils/logger.py — Structured logging for profiling and debugging.
"""

import time


class Logger:
    """Simple structured logger with timing support."""

    def __init__(self, name: str = "ThereSyn", enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def info(self, msg: str):
        if self.enabled:
            print(f"[{self.name}] {msg}")

    def warn(self, msg: str):
        if self.enabled:
            print(f"[{self.name} WARNING] {msg}")

    def error(self, msg: str):
        print(f"[{self.name} ERROR] {msg}")

    def latency_report(self, report: dict):
        """Print a formatted latency report."""
        if not self.enabled or not report:
            return
        lines = [f"[{self.name} LATENCY]"]
        for stage, stats in report.items():
            if stats is None:
                continue
            lines.append(
                f"  {stage:15s}: avg={stats['avg_ms']:5.1f}ms  "
                f"max={stats['max_ms']:5.1f}ms  p95={stats['p95_ms']:5.1f}ms"
            )
        print("\n".join(lines))
