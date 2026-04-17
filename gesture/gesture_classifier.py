"""
gesture/gesture_classifier.py — ML-based gesture classifier.

Collects hand landmark samples, trains a lightweight classifier (sklearn MLP),
and exports to ONNX for fast inference.

Gestures:
  - rest: Hand open, flat (no interaction)
  - engage: Pinch (thumb + index close together)
  - stop: Closed fist
  - wave: Open palm moving laterally
"""

import numpy as np
from config import GESTURE_MODEL_PATH, GESTURE_LABELS, GESTURE_CONFIDENCE_THRESHOLD
import os


class GestureClassifier:
    """ML gesture classifier using ONNX runtime for inference."""

    def __init__(self, model_path: str = GESTURE_MODEL_PATH):
        self.model_path = model_path
        self._session = None
        self._labels = GESTURE_LABELS
        self.confidence_threshold = GESTURE_CONFIDENCE_THRESHOLD
        self._load_model()

    def _load_model(self):
        """Load ONNX model if available."""
        if not os.path.exists(self.model_path):
            return
        try:
            import onnxruntime as ort
            import onnx
            self._session = ort.InferenceSession(self.model_path)

            # Read labels from model metadata (written during training)
            onnx_model = onnx.load(self.model_path)
            metadata = {m.key: m.value for m in onnx_model.metadata_props}
            if "gesture_labels" in metadata:
                import json
                self._labels = json.loads(metadata["gesture_labels"])
                print(f"[ML] Loaded labels from model: {self._labels}")

            print(f"[ML] Loaded model from {self.model_path}")
        except ImportError:
            print("[ML] onnxruntime not installed — running without ML")
        except Exception as e:
            print(f"[ML] Failed to load model: {e}")

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

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

        landmarks = hands_data[0]["landmarks"]
        input_data = []
        for x, y, z in landmarks:
            input_data.extend([x, y, z])

        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)

        try:
            outputs = self._session.run(None, {"input": input_array})

            # outputs[0] = predicted class labels (1D array)
            # outputs[1] = probabilities (2D array, shape [1, n_classes])
            #   OR outputs[1] = zipmap list (if zipmap not disabled)
            if len(outputs) < 2:
                return None

            pred_label = int(outputs[0][0])
            prob_data = outputs[1][0]

            if isinstance(prob_data, dict):
                # Zipmap format: {0: prob, 1: prob, ...}
                confidence = float(prob_data.get(pred_label, 0.0))
            elif isinstance(prob_data, np.ndarray):
                # Raw probability array
                confidence = float(prob_data[pred_label])
            else:
                confidence = float(prob_data)

            if confidence >= self.confidence_threshold:
                if pred_label < len(self._labels):
                    return {
                        "gesture": self._labels[pred_label],
                        "confidence": confidence,
                    }
            return None
        except Exception as e:
            print(f"[ML] Inference error: {e}")
            return None


class GestureDataCollector:
    """Utility for recording gesture training samples."""

    def __init__(self, labels: list = None):
        self._labels = labels or GESTURE_LABELS
        self._samples = {label: [] for label in self._labels}
        self._current_label = None

    def set_label(self, label: str):
        if label in self._labels:
            self._current_label = label
        else:
            print(f"[ML] Unknown label: {label}. Valid: {self._labels}")

    def record_sample(self, hands_data: list):
        """Record current hand landmarks as a training sample."""
        if self._current_label is None:
            print("[ML] No label set. Call set_label() first.")
            return
        if not hands_data:
            return

        landmarks = hands_data[0]["landmarks"]
        flat = []
        for x, y, z in landmarks:
            flat.extend([x, y, z])
        self._samples[self._current_label].append(flat)

    def get_sample_counts(self) -> dict:
        return {label: len(samples) for label, samples in self._samples.items()}

    def get_dataset(self) -> tuple:
        """Return (X, y) arrays ready for sklearn."""
        X_list = []
        y_list = []
        for label, samples in self._samples.items():
            for sample in samples:
                X_list.append(sample)
                y_list.append(self._labels.index(label))

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    def save(self, path: str):
        """Save collected samples to numpy file."""
        X, y = self.get_dataset()
        if len(X) == 0:
            print("[ML] No samples to save.")
            return
        np.savez(path, X=X, y=y, labels=np.array(self._labels))
        print(f"[ML] Saved {len(X)} samples ({len(self._labels)} classes) to {path}.npz")


def train_classifier(X: np.ndarray, y: np.ndarray, labels: list,
                     hidden_sizes=(128, 64), max_iter=500,
                     output_path: str = GESTURE_MODEL_PATH) -> bool:
    """
    Train an MLP classifier and export to ONNX.

    Args:
        X: Feature matrix (n_samples, 63)
        y: Label indices (n_samples,)
        labels: List of class names
        hidden_sizes: MLP hidden layer sizes
        max_iter: Training iterations
        output_path: Where to save the ONNX model

    Returns:
        True if training and export succeeded.
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except ImportError as e:
        print(f"[ML] sklearn not available: {e}. Skipping training.")
        return False

    # Check minimum samples
    min_per_class = 10
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        if count < min_per_class:
            print(f"[ML] Class {labels[cls]} has only {count} samples (need {min_per_class}). Skipping training.")
            return False

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=0.001,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"[ML] Training accuracy: {train_acc:.3f}")
    print(f"[ML] Test accuracy:     {test_acc:.3f}")

    if len(labels) <= 10:
        y_pred = clf.predict(X_test)
        print("[ML] Classification report:")
        print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # Export to ONNX
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
        # Disable zipmap to get probability arrays
        onnx_model = convert_sklearn(clf, initial_types=initial_type, options={id(clf): {"zipmap": False}})

        # Validate model
        onnx.checker.check_model(onnx_model)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Verify loaded model produces same results
        sess = ort.InferenceSession(output_path)
        test_input = X_test[:1].astype(np.float32)
        onnx_outputs = sess.run(None, {"input": test_input})
        onnx_pred = onnx_outputs[0]
        skl_pred = clf.predict_proba(X_test[:1])
        # We expect probability arrays; compare shapes and closeness
        if onnx_pred.shape == skl_pred.shape:
            if not np.allclose(onnx_pred, skl_pred, atol=1e-4):
                print("[ML] WARNING: ONNX predictions differ from sklearn (tolerance exceeded)")
        else:
            print("[ML] WARNING: ONNX output shape differs from sklearn")

        print(f"[ML] Model exported to {output_path}")
        return True

    except ImportError as e:
        print(f"[ML] ONNX export failed (missing package): {e}")
        print("[ML] Install with: pip install skl2onnx onnx onnxruntime")
        return False
    except Exception as e:
        print(f"[ML] ONNX export failed: {e}")
        return False
