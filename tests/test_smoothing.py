from utils.smoothing import LandmarkSmoother


def test_smoother_initialization_and_update():
    s = LandmarkSmoother(alpha=0.5)
    hands = [{"type": "Right", "landmarks": [(0.1, 0.1, 0.0)] * 21, "raw_landmarks": None}]
    out1 = s.smooth(hands)
    assert len(out1) == 1

    hands2 = [{"type": "Right", "landmarks": [(0.2, 0.2, 0.0)] * 21, "raw_landmarks": None}]
    out2 = s.smooth(hands2)
    assert out2[0]["landmarks"][0] != out1[0]["landmarks"][0]
