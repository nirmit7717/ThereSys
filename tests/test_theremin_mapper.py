from gesture.theremin_mapper import ThereminMapper


def make_landmarks(wrist_x, wrist_y):
    # 21 landmarks, each (x,y,z)
    landmarks = [(wrist_x, wrist_y, 0.0)] + [(0.5, 0.5, 0.0)] * 20
    return landmarks


def test_pitch_and_volume_extremes():
    tm = ThereminMapper(freq_min=100.0, freq_max=800.0, vol_min=0.0, vol_max=1.0)
    lm_left = make_landmarks(0.0, 1.0)  # leftmost, bottom
    out = tm.map_hand(lm_left, engaged=True)
    assert out["pitch"] >= 100.0 and out["pitch"] <= 800.0

    lm_high = make_landmarks(0.5, 0.0)
    out_high = tm.map_hand(lm_high, engaged=True)
    lm_low = make_landmarks(0.5, 1.0)
    out_low = tm.map_hand(lm_low, engaged=True)
    # higher hand should produce >= volume than lower hand (considering inversion)
    assert out_high["volume"] >= out_low["volume"]
