"""Tests for Phase 1 modules: smoothing, pinch, theremin_mapper, dsp"""
import math
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_smoothing():
    from utils.smoothing import LandmarkSmoother
    smoother = LandmarkSmoother(alpha=0.5)

    # Single frame — should initialize directly
    hands_data = [{
        "type": "Right",
        "landmarks": [(0.5, 0.5, 0.0)] * 21,
        "raw_landmarks": None,
    }]
    result = smoother.smooth(hands_data)
    assert len(result) == 1
    assert result[0]["landmarks"] == [(0.5, 0.5, 0.0)] * 21
    print("  smoothing: init pass")

    # Second frame with offset — should blend
    hands_data2 = [{
        "type": "Right",
        "landmarks": [(0.6, 0.6, 0.1)] * 21,
        "raw_landmarks": None,
    }]
    result2 = smoother.smooth(hands_data2)
    x, y, z = result2[0]["landmarks"][0]
    # alpha=0.5: 0.5 + 0.5*(0.6-0.5) = 0.55
    assert abs(x - 0.55) < 0.001, f"Expected 0.55, got {x}"
    assert abs(y - 0.55) < 0.001, f"Expected 0.55, got {y}"
    assert abs(z - 0.05) < 0.001, f"Expected 0.05, got {z}"
    print("  smoothing: EMA blend pass")

    # Two hands
    hands_data3 = [
        {"type": "Left", "landmarks": [(0.1, 0.1, 0.0)] * 21, "raw_landmarks": None},
        {"type": "Right", "landmarks": [(0.9, 0.9, 0.0)] * 21, "raw_landmarks": None},
    ]
    result3 = smoother.smooth(hands_data3)
    assert len(result3) == 2
    assert result3[0]["type"] == "Left"
    assert result3[1]["type"] == "Right"
    print("  smoothing: multi-hand pass")

    # Hand lost — stale cleanup
    hands_data4 = [
        {"type": "Right", "landmarks": [(0.9, 0.9, 0.0)] * 21, "raw_landmarks": None},
    ]
    result4 = smoother.smooth(hands_data4)
    assert len(result4) == 1
    print("  smoothing: hand loss cleanup pass")

    print("[PASS] LandmarkSmoother")
    return True


def test_pinch_detector():
    from gesture.pinch_detector import PinchDetector
    detector = PinchDetector(threshold=0.05)

    # No hands — empty result
    result = detector.detect([])
    assert result == {}
    print("  pinch: empty hands pass")

    # Hand with thumb+index far apart — not pinching
    landmarks_open = [(0.0, 0.0, 0.0)] * 21
    landmarks_open[4] = (0.1, 0.0, 0.0)   # thumb tip far right
    landmarks_open[8] = (0.3, 0.0, 0.0)   # index tip further right
    hands = [{"type": "Right", "landmarks": landmarks_open, "raw_landmarks": None}]
    result = detector.detect(hands)
    assert result[0]["pinching"] == False
    assert result[0]["just_pinched"] == False
    print("  pinch: open hand pass")

    # Hand with thumb+index close — pinching
    landmarks_pinch = [(0.0, 0.0, 0.0)] * 21
    landmarks_pinch[4] = (0.25, 0.35, 0.0)
    landmarks_pinch[8] = (0.26, 0.35, 0.0)
    hands = [{"type": "Right", "landmarks": landmarks_pinch, "raw_landmarks": None}]
    result = detector.detect(hands)
    assert result[0]["pinching"] == True
    assert result[0]["just_pinched"] == True
    print("  pinch: pinch detected pass")

    # Same pinch next frame — not just_pinched anymore
    result2 = detector.detect(hands)
    assert result2[0]["pinching"] == True
    assert result2[0]["just_pinched"] == False
    print("  pinch: sustained pinch (no re-trigger) pass")

    # Release
    hands_open = [{"type": "Right", "landmarks": landmarks_open, "raw_landmarks": None}]
    result3 = detector.detect(hands_open)
    assert result3[0]["pinching"] == False
    assert result3[0]["just_released"] == True
    print("  pinch: release detected pass")

    # Center calculation
    assert abs(result[0]["center"][0] - 0.255) < 0.01
    assert abs(result[0]["center"][1] - 0.35) < 0.01
    print("  pinch: center calc pass")

    print("[PASS] PinchDetector")
    return True


def test_theremin_mapper():
    from gesture.theremin_mapper import ThereminMapper
    mapper = ThereminMapper()

    # Hand at center, not engaged → volume 0
    landmarks = [(0.5, 0.5, 0.0)] * 21
    result = mapper.map_hand(landmarks, engaged=False)
    assert result["engaged"] == False
    assert result["volume"] == 0.0
    print("  mapper: disengaged → vol=0 pass")

    # Engaged at center — multiple calls to converge through internal smoothing
    for _ in range(10):
        result = mapper.map_hand(landmarks, engaged=True)
    assert result["engaged"] == True
    assert result["volume"] > 0.3
    expected_mid = 2 ** ((mapper._log_freq_min + mapper._log_freq_max) / 2)
    assert abs(result["pitch"] - expected_mid) < 50, f"Expected ~{expected_mid:.1f}, got {result['pitch']:.1f}"
    print(f"  mapper: center engage → pitch={result['pitch']:.1f}Hz pass")

    # Left edge → low freq
    landmarks_left = [(0.0, 0.5, 0.0)] * 21
    for _ in range(10):
        result_left = mapper.map_hand(landmarks_left, engaged=True)
    assert result_left["pitch"] < 200, f"Left edge should be low, got {result_left['pitch']:.1f}"
    print(f"  mapper: left edge → pitch={result_left['pitch']:.1f}Hz pass")

    # Right edge → high freq
    landmarks_right = [(1.0, 0.5, 0.0)] * 21
    for _ in range(10):
        result_right = mapper.map_hand(landmarks_right, engaged=True)
    assert result_right["pitch"] > 800, f"Right edge should be high, got {result_right['pitch']:.1f}"
    print(f"  mapper: right edge → pitch={result_right['pitch']:.1f}Hz pass")

    # Top of frame → higher volume (Y inverted)
    landmarks_top = [(0.5, 0.0, 0.0)] * 21
    for _ in range(10):
        result_top = mapper.map_hand(landmarks_top, engaged=True)
    assert result_top["volume"] > 0.7, f"Top should be loud, got {result_top['volume']:.2f}"
    print(f"  mapper: top → vol={result_top['volume']:.2f} pass")

    # Bottom → lower volume
    landmarks_bottom = [(0.5, 1.0, 0.0)] * 21
    for _ in range(10):
        result_bottom = mapper.map_hand(landmarks_bottom, engaged=True)
    assert result_bottom["volume"] < 0.3, f"Bottom should be quiet, got {result_bottom['volume']:.2f}"
    print(f"  mapper: bottom → vol={result_bottom['volume']:.2f} pass")

    # Filter cutoff: open hand (fingers spread) vs fist
    landmarks_open = [(0.5, 0.5, 0.0)] * 21
    landmarks_open[4] = (0.7, 0.3, 0.0)   # thumb far
    landmarks_open[8] = (0.3, 0.2, 0.0)   # index far
    landmarks_open[12] = (0.3, 0.5, 0.0)  # middle far
    landmarks_open[16] = (0.5, 0.7, 0.0)  # ring far
    landmarks_open[20] = (0.7, 0.8, 0.0)  # pinky far
    result_open = mapper.map_hand(landmarks_open, engaged=True)

    landmarks_fist = [(0.5, 0.5, 0.0)] * 21
    landmarks_fist[4] = (0.55, 0.42, 0.0)
    landmarks_fist[8] = (0.48, 0.40, 0.0)
    landmarks_fist[12] = (0.52, 0.55, 0.0)
    landmarks_fist[16] = (0.55, 0.58, 0.0)
    landmarks_fist[20] = (0.58, 0.60, 0.0)
    for _ in range(10):
        result_fist = mapper.map_hand(landmarks_fist, engaged=True)

    assert result_open["filter_cutoff"] > result_fist["filter_cutoff"], \
        f"Open hand ({result_open['filter_cutoff']:.2f}) should be brighter than fist ({result_fist['filter_cutoff']:.2f})"
    print(f"  mapper: open hand cutoff={result_open['filter_cutoff']:.2f}, fist cutoff={result_fist['filter_cutoff']:.2f} pass")

    # Reset
    mapper.reset()
    result_reset = mapper.map_hand(landmarks, engaged=False)
    assert result_reset["volume"] == 0.0, f"Reset+disengaged vol should be 0, got {result_reset['volume']:.2f}"
    assert result_reset["engaged"] == False
    print("  mapper: reset pass")

    print("[PASS] ThereminMapper")
    return True


def test_oscillator():
    from dsp.oscillator import Oscillator
    osc = Oscillator(sample_rate=44100)

    # Sine wave (default volume=0.5)
    wave = osc.generate("sine", 440.0, 0.1)
    assert len(wave) == 4410, f"Expected 4410 samples, got {len(wave)}"
    assert np.max(np.abs(wave)) <= 1.0, "Wave should be normalized"
    assert 0.49 < np.max(np.abs(wave)) <= 0.51, f"Sine peak with vol=0.5 should be ~0.5, got {np.max(np.abs(wave))}"
    # Full volume sine
    wave_full = osc.generate("sine", 440.0, 0.1, volume=1.0)
    assert np.max(np.abs(wave_full)) > 0.99, f"Full vol sine peak should be ~1.0, got {np.max(np.abs(wave_full))}"
    print("  oscillator: sine pass")

    # Sawtooth
    wave_saw = osc.generate("sawtooth", 440.0, 0.1)
    assert len(wave_saw) == 4410
    assert np.max(np.abs(wave_saw)) <= 1.0
    print("  oscillator: sawtooth pass")

    # Square
    wave_sq = osc.generate("square", 440.0, 0.1)
    assert len(wave_sq) == 4410
    print("  oscillator: square pass")

    # Triangle
    wave_tri = osc.generate("triangle", 440.0, 0.1)
    assert len(wave_tri) == 4410
    print("  oscillator: triangle pass")

    # Invalid waveform → falls back to sine
    wave_inv = osc.generate("invalid", 440.0, 0.1)
    assert len(wave_inv) == 4410
    print("  oscillator: invalid fallback pass")

    # Volume control
    wave_quiet = osc.generate("sine", 440.0, 0.1, volume=0.25)
    assert 0.24 < np.max(np.abs(wave_quiet)) <= 0.251, f"Quiet wave max should be ~0.25, got {np.max(np.abs(wave_quiet))}"
    print("  oscillator: volume control pass")

    # Frequency accuracy: generate sine at 1000Hz, check zero crossings
    wave_1k = osc.generate("sine", 1000.0, 1.0)
    sr = 44100
    # Period = 44.1 samples. Check distance between first two positive zero crossings
    crossings = np.where(np.diff(np.sign(wave_1k)) > 0)[0]
    if len(crossings) >= 2:
        period = crossings[1] - crossings[0]
        expected_period = sr / 1000.0
        assert abs(period - expected_period) < 2, f"Expected period ~{expected_period:.1f}, got {period}"
        print(f"  oscillator: freq accuracy (1000Hz, period={period:.1f} samples) pass")
    else:
        print("  oscillator: freq accuracy SKIP (not enough crossings)")

    # mixer output
    stereo = osc.generate_for_mixer("sine", 440.0, 0.1)
    assert stereo.shape == (4410, 2), f"Expected (4410, 2), got {stereo.shape}"
    assert stereo.dtype == np.int16
    print("  oscillator: mixer output pass")

    print("[PASS] Oscillator")
    return True


def test_envelope():
    from dsp.envelope import ADSREnvelope
    env = ADSREnvelope(attack=0.01, decay=0.01, sustain_level=0.5, release=0.01, sample_rate=1000)

    # 1 second signal at full amplitude
    wave = np.ones(1000)
    shaped = env.apply(wave)

    # Start should be near 0
    assert shaped[0] < 0.1, f"Attack start should be near 0, got {shaped[0]}"
    # After attack+decay, should be at sustain level
    assert abs(shaped[20] - 0.5) < 0.1, f"Sustain should be ~0.5, got {shaped[20]}"
    # End should be near 0 (release)
    assert shaped[-1] < 0.1, f"Release end should be near 0, got {shaped[-1]}"
    print("  envelope: ADSR shape pass")

    # No clipping
    assert np.max(np.abs(shaped)) <= 1.0, "Envelope should not exceed 1.0"
    print("  envelope: no clipping pass")

    print("[PASS] ADSREnvelope")
    return True


def test_filter():
    from dsp.filter import LowpassFilter
    filt = LowpassFilter(sample_rate=44100)

    # High-frequency signal fully open (cutoff=1.0) should pass through
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    high_freq = np.sin(2 * np.pi * 5000 * t)
    filtered_open = filt.apply(high_freq.copy(), cutoff_normalized=1.0)
    # Should still have significant energy
    rms_open = np.sqrt(np.mean(filtered_open ** 2))
    assert rms_open > 0.1, f"Open filter should pass signal, RMS={rms_open}"
    print(f"  filter: open pass-through RMS={rms_open:.3f} pass")

    # Fully closed (cutoff=0.0) should heavily attenuate
    filtered_closed = filt.apply(high_freq, cutoff_normalized=0.0)
    rms_closed = np.sqrt(np.mean(filtered_closed ** 2))
    assert rms_closed < rms_open, f"Closed filter should attenuate: open={rms_open:.3f}, closed={rms_closed:.3f}"
    print(f"  filter: closed attenuation RMS={rms_closed:.3f} pass")

    print("[PASS] LowpassFilter")
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0

    for test_fn in [test_smoothing, test_pinch_detector, test_theremin_mapper,
                    test_oscillator, test_envelope, test_filter]:
        print(f"\n{'='*50}")
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
