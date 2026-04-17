import time

from engine.audio_engine import AudioEngine


def test_audio_engine_queue_headless():
    # Create headless audio engine (no pygame/numpy usage)
    eng = AudioEngine(enable_audio=False)

    try:
        # Send update and engage commands
        eng.update_theremin(frequency=440.0, volume=0.7, filter_cutoff=0.5, waveform='sine')
        eng.theremin_engage()

        # Allow background thread to process
        time.sleep(0.1)

        # Check that internal state was updated by the audio thread
        assert abs(eng._freq - 440.0) < 1e-6
        assert abs(eng._vol - 0.7) < 1e-6
        assert eng._engaged is True

        # Disengage
        eng.theremin_disengage()
        time.sleep(0.05)
        assert eng._engaged is False
    finally:
        eng.quit()
