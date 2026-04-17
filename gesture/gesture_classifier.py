"""
gesture/gesture_classifier.py — ML-based gesture classifier.

Collects hand landmark samples, trains a lightweight classifier (sklearn MLP),
and exports to ONNX for fast inference.

Supported gestures:
  - piano_rest: Hand open, flat (not playing)
  - theremin_engage: Pinch (thumb + index close)
  - octave_up: Two fingers pointing up
  - octave_down: Two fingers pointing down
  - play: Open palm facing camera
  - stop: Closed fist

TODO (Collaborator A): Implement data collection, training, and inference.
"""

from config import GESTURE_MODEL_PATH, GESTURE_LABELS, GESTURE_CONFIDENCE_THRESHOLD


class GestureClassifier:
    """ML gesture classifier using ONNX runtime for inference."""

    def __init__(self, model_path: str = GESTURE_MODEL_PATH):
        self.model_path = model_path
        self._session = None
        self.labels = GESTURE_LABELS
        self.confidence_threshold = GESTURE_CONFIDENCE_THRESHOLD
        self._load_model()

    def _load_model(self):
        """Load ONNX model if available."""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path)
            print(f"[ML] Loaded model from {self.model_path}")
        except Exception:
            print("[ML] No model found — running in rule-based fallback mode.")

    def classify(self, hands_data: list) -> dict | None:
        """
        Classify current hand gesture.

        Args:
            hands_data: Smoothed hand data.

        Returns:
            {"gesture": str, "confidence": float} or None if no model/no detection.
        """
        if self._session is None:
            return None

        if not hands_data:
            return None

        # Use first hand's landmarks as input
        landmarks = hands_data[0]["landmarks"]
        # Flatten to 1D array: [x0, y0, z0, x1, y1, z1, ...]
        input_data = []
        for x, y, z in landmarks:
            input_data.extend([x, y, z])

        import numpy as np
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)

        try:
            outputs = self._session.run(None, {"input": input_array})
            probabilities = outputs[0][0]
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])

            if confidence >= self.confidence_threshold:
                return {
                    "gesture": self.labels[best_idx],
                    "confidence": confidence,
                }
            return None
        except Exception as e:
            print(f"[ML] Inference error: {e}")
            return None


class GestureDataCollector:
    """Utility for recording gesture training samples."""

    def __init__(self, labels: list = GESTURE_LABELS):
        self.labels = labels
        self._samples = {label: [] for label in labels}
        self._current_label = None

    def set_label(self, label: str):
        self._current_label = label

    def record_sample(self, hands_data: list):
        """Record current hand landmarks as a training sample."""
        if self._current_label is None or not hands_data:
            return
        landmarks = hands_data[0]["landmarks"]
        flat = []
        for x, y, z in landmarks:
            flat.extend([x, y, z])
        self._samples[self._current_label].append(flat)

    def get_samples(self) -> dict:
        return self._samples

    def save(self, path: str):
        """Save collected samples to numpy file."""
        import numpy as np
        import json

        data = {}
        for label, samples in self._samples.items():
            if samples:
                data[label] = np.array(samples)

        np.savez(path, **data)
        with open(path + ".labels.json", "w") as f:
            json.dump(list(data.keys()), f)
        print(f"[ML] Saved {sum(len(v) for v in self._samples.values())} samples to {path}")
