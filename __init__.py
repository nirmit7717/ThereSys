"""
ThereSyn — AR Theremin-Synthesizer

Real-time gesture-based instrument combining:
  - Spatial Theremin mode (continuous pitch + amplitude control)
  - Piano mode (discrete key triggers via hitbox collision)
  - ML gesture classifier for mode switching
  - Latency-aware architecture with predictive rendering
  - MIDI output for DAW integration

Architecture Overview:
  Camera → Vision → Gesture Engine → [Theremin Mode | Piano Mode] → DSP Engine → Audio Output → UI Overlay
"""

__version__ = "0.1.0"
