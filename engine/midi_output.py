"""
engine/midi_output.py — MIDI stream output for DAW integration.

Sends note_on, note_off, pitch_bend, and control_change messages
via rtmidi to any connected MIDI port or virtual MIDI device.
"""

import sys
from config import MIDI_OUTPUT_NAME, MIDI_ENABLED


class MIDIOutput:
    """MIDI output handler using rtmidi (python-rtmidi)."""

    def __init__(self, enabled: bool = MIDI_ENABLED, output_name: str = MIDI_OUTPUT_NAME):
        self.enabled = enabled
        self._output = None
        self._api = None  # Track which API style we're using

        if self.enabled:
            try:
                import rtmidi
                self._midi_out = rtmidi.MidiOut()
                self._open_port(self._midi_out, output_name)
                self._output = self._midi_out
                self._api = "python-rtmidi"
            except ImportError:
                print("[MIDI] python-rtmidi not installed. MIDI disabled.")
                self.enabled = False
            except Exception as e:
                print(f"[MIDI] Failed to open MIDI output: {e}")
                self.enabled = False

    @staticmethod
    def _open_port(midi_out, name: str):
        """Open a MIDI port. Tries virtual port first (macOS/Linux),
        falls back to listing and opening an available port (Windows)."""
        if sys.platform == "win32":
            # Windows doesn't support virtual ports — open first available output port
            ports = midi_out.get_ports()
            if ports:
                midi_out.open_port(0)
                print(f"[MIDI] Opened port: {ports[0]}")
            else:
                # No ports available — create a virtual one anyway (some drivers support it)
                try:
                    midi_out.open_virtual_port(name)
                    print(f"[MIDI] Virtual port '{name}' opened (Windows fallback).")
                except Exception:
                    print("[MIDI] No MIDI output ports found and virtual port not supported.")
                    print("[MIDI] Install a virtual MIDI driver like 'loopMIDI' for Windows MIDI output.")
                    raise RuntimeError("No MIDI ports available")
        else:
            # macOS / Linux — use virtual port
            midi_out.open_virtual_port(name)
            print(f"[MIDI] Virtual port '{name}' opened.")

    def note_on(self, note: int, velocity: int = 127, channel: int = 0):
        """Send MIDI note_on message."""
        if not self.enabled:
            return
        msg = [0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F]
        self._output.send_message(msg)

    def note_off(self, note: int, velocity: int = 0, channel: int = 0):
        """Send MIDI note_off message."""
        if not self.enabled:
            return
        msg = [0x80 | (channel & 0x0F), note & 0x7F, velocity & 0x7F]
        self._output.send_message(msg)

    def pitch_bend(self, value: int, channel: int = 0):
        """
        Send MIDI pitch_bend message.

        Args:
            value: 0-16383, 8192 = center (no bend).
            channel: MIDI channel.
        """
        if not self.enabled:
            return
        lsb = value & 0x7F
        msb = (value >> 7) & 0x7F
        msg = [0xE0 | (channel & 0x0F), lsb, msb]
        self._output.send_message(msg)

    def control_change(self, cc: int, value: int, channel: int = 0):
        """Send MIDI CC message."""
        if not self.enabled:
            return
        msg = [0xB0 | (channel & 0x0F), cc & 0x7F, value & 0x7F]
        self._output.send_message(msg)

    def set_instrument(self, program: int, channel: int = 0):
        """Send MIDI program change (instrument selection)."""
        if not self.enabled:
            return
        msg = [0xC0 | (channel & 0x0F), program & 0x7F]
        self._output.send_message(msg)

    def close(self):
        """Close MIDI port."""
        if self._output:
            self._output.close_port()
            del self._output

    @staticmethod
    def note_name_to_midi(note_name: str) -> int:
        """Convert note name like 'C4' to MIDI note number."""
        note_map = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
                     "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

        if note_name[1] == "#":
            name_part = note_name[:2]
            octave_part = note_name[2:]
        else:
            name_part = note_name[0]
            octave_part = note_name[1:]

        if name_part not in note_map:
            return 60  # default C4

        try:
            octave = int(octave_part)
        except ValueError:
            return 60

        return 12 * (octave + 1) + note_map[name_part]

    @staticmethod
    def freq_to_midi(freq: float) -> int:
        """Convert frequency (Hz) to nearest MIDI note number.

        Formula: midi = 69 + 12 * log2(freq / 440.0)
        """
        import math

        if freq <= 0:
            return 60
        midi = 69 + 12 * math.log2(freq / 440.0)
        return int(round(midi))

    @staticmethod
    def freq_to_pitch_bend(freq: float, base_note: int) -> int:
        """
        Convert frequency to MIDI pitch_bend value for microtonal control.

        Args:
            freq: Target frequency in Hz.
            base_note: Integer MIDI note to bend from.

        Returns:
            Pitch bend value 0-16383 (8192 = center).

        Assumes pitch bend range is ±2 semitones (200 cents). Adjust if DAW uses a different range.
        """
        import math

        if freq <= 0 or base_note < 0 or base_note > 127:
            return 8192
        base_freq = 440.0 * (2.0 ** ((base_note - 69) / 12.0))
        if base_freq <= 0:
            return 8192
        # cents difference: 1200 * log2(freq / base_freq)
        cents = 1200.0 * math.log2(freq / base_freq)
        # Map cents to pitch_bend range using configured semitone range
        from config import MIDI_PITCH_BEND_RANGE_SEMITONES
        cents_range = MIDI_PITCH_BEND_RANGE_SEMITONES * 100.0
        bend = int(8192 + (cents / cents_range) * 8191)
        return max(0, min(16383, bend))
