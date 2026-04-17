from gesture.gesture_classifier import GestureClassifier


def test_gesture_classifier_fallback():
    gc = GestureClassifier()
    # If ONNX not available, classifier should initialize but session be None
    if gc._session is None:
        res = gc.classify([])
        assert res is None
    else:
        # If model present, ensure classify doesn't crash on empty input
        res = gc.classify([])
        assert res is None or isinstance(res, dict)
