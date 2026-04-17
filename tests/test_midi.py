import math
from engine.midi_output import MIDIOutput


def test_freq_to_midi_and_pitch_bend():
    # A4 = 440 Hz -> MIDI 69
    assert MIDIOutput.freq_to_midi(440.0) == 69
    # A5 = 880 Hz -> 81
    assert MIDIOutput.freq_to_midi(880.0) == 81
    # Middle C ~261.6256 -> 60
    assert MIDIOutput.freq_to_midi(261.625565) == 60

    base = MIDIOutput.freq_to_midi(440.0)
    assert MIDIOutput.freq_to_pitch_bend(440.0, base) == 8192

    # small upward semitone: A4 * 2^(1/12)
    semitone_up = 440.0 * (2 ** (1/12))
    bend = MIDIOutput.freq_to_pitch_bend(semitone_up, 69)
    assert 8192 < bend <= 16383
