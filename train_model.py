"""
train_model.py — Train the gesture classifier and export to ONNX.

Usage:
    # 1. Collect samples (run with webcam, press keys to label):
    python train_model.py --collect

    # 2. Train from collected samples:
    python train_model.py --train

    # 3. Verify exported model:
    python train_model.py --verify
"""

import argparse
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GESTURE_MODEL_PATH, GESTURE_LABELS, GESTURE_CONFIDENCE_THRESHOLD
from gesture.gesture_classifier import GestureClassifier, GestureDataCollector


def train(samples_path: str, output_path: str = GESTURE_MODEL_PATH,
          hidden_sizes=(128, 64), max_iter=500):
    """Train an MLP classifier from saved samples and export to ONNX."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Load samples
    if not os.path.exists(samples_path):
        print(f"[Train] No samples file found at {samples_path}")
        print("[Train] Collect samples first: python train_model.py --collect")
        return False

    data = np.load(samples_path)
    labels_list = list(data.keys())
    print(f"[Train] Found classes: {labels_list}")

    X_list = []
    y_list = []

    for label_idx, label in enumerate(labels_list):
        samples = data[label]
        print(f"  {label}: {len(samples)} samples")
        for sample in samples:
            X_list.append(sample)
            y_list.append(label_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"[Train] Total: {len(X)} samples, {len(labels_list)} classes, {X.shape[1]} features")

    # Check minimum per class
    unique, counts = np.unique(y, return_counts=True)
    min_per_class = 10
    for cls, count in zip(unique, counts):
        if count < min_per_class:
            print(f"[Train] WARNING: Class '{labels_list[cls]}' has only {count} samples (recommend {min_per_class}+)")

    if len(unique) < 2:
        print("[Train] Need at least 2 classes to train. Collect more gesture types.")
        return False

    # Split
    test_size = 0.2 if len(X) >= 40 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(unique) > 1 else None
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

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"[Train] Training accuracy: {train_acc:.3f}")
    print(f"[Train] Test accuracy:     {test_acc:.3f}")

    if len(labels_list) <= 10:
        y_pred = clf.predict(X_test)
        print("[Train] Classification report:")
        print(classification_report(y_test, y_pred, target_names=labels_list, zero_division=0))

    # Export to ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
        # Disable zipmap output — get raw probability arrays instead of dicts
        onnx_model = convert_sklearn(
            clf,
            initial_types=initial_type,
            options={id(clf): {"zipmap": False}},
        )

        # Embed labels as model metadata so classifier can read them
        import json
        labels_entry = onnx_model.metadata_props.add()
        labels_entry.key = "gesture_labels"
        labels_entry.value = json.dumps(labels_list)

        # Validate
        onnx.checker.check_model(onnx_model)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[Train] Model exported to {output_path}")
        print(f"[Train] Labels embedded: {labels_list}")

        # Verify
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        test_input = X_test[:1].astype(np.float32)
        onnx_outputs = sess.run(None, {"input": test_input})
        skl_pred = clf.predict(X_test[:1])
        onnx_pred = onnx_outputs[0]
        if np.array_equal(onnx_pred, skl_pred):
            print("[Train] ONNX export verified — predictions match sklearn")
        else:
            print(f"[Train] WARNING: ONNX pred={onnx_pred} vs sklearn pred={skl_pred}")

        return True

    except ImportError as e:
        print(f"[Train] ONNX export skipped (missing package): {e}")
        print("[Train] Install with: pip install skl2onnx onnx onnxruntime")
        return False
    except Exception as e:
        print(f"[Train] ONNX export failed: {e}")
        return False


def verify(model_path: str = GESTURE_MODEL_PATH):
    """Verify the exported ONNX model works."""
    clf = GestureClassifier(model_path=model_path)
    if not hasattr(clf, '_session') or clf._session is None:
        print("[Verify] No model loaded.")
        return

    print("[Verify] Model loaded successfully.")
    print(f"[Verify] Labels: {clf.labels}")

    # Test with dummy data
    dummy = [(0.5, 0.5, 0.0)] * 21
    result = clf.classify([{"landmarks": dummy}])
    if result:
        print(f"[Verify] Dummy input → {result['gesture']} (conf={result['confidence']:.2f})")
    else:
        print("[Verify] Dummy input → no prediction (below threshold)")
    print("[Verify] OK")


def collect(samples_path: str = "assets/samples.npz"):
    """Interactive data collection mode with webcam."""
    import cv2
    import mediapipe as mp
    from vision.camera import Camera
    from vision.hand_tracker import HandTracker
    from utils.smoothing import LandmarkSmoother

    camera = Camera(0, 640, 480)
    tracker = HandTracker()
    smoother = LandmarkSmoother()
    collector = GestureDataCollector()

    print("\n=== Gesture Data Collection ===")
    print(f"Labels: {GESTURE_LABELS}")
    print("Press number keys (1-7) to set label, SPACE to record sample, S to save, Q to quit\n")

    label_map = {str(i + 1): label for i, label in enumerate(GESTURE_LABELS)}
    current_label = None
    sample_count = 0

    while True:
        frame = camera.read_frame()
        if frame is None:
            break

        hands = tracker.process_frame(frame)
        smoothed = smoother.smooth(hands)

        # Draw landmarks
        for hand in hands:
            if "raw_landmarks" in hand:
                # Use MediaPipe drawing utilities
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand["raw_landmarks"],
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        # Status overlay
        status = f"Label: {current_label or 'none'} | Samples: {sample_count} | Press 1-{len(GESTURE_LABELS)} to set label"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Collect Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            collector.save(samples_path)
            print(f"[Collect] Saved to {samples_path}")
        elif key == ord(' '):
            if current_label and smoothed:
                collector.record_sample(smoothed)
                sample_count += 1
                print(f"  Recorded '{current_label}' (#{sample_count})")
        elif chr(key) in label_map:
            current_label = label_map[chr(key)]
            collector.set_label(current_label)
            print(f"  Label set to: {current_label}")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\n[Collect] Total: {sample_count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThereSyn ML Training")
    parser.add_argument("--collect", action="store_true", help="Collect gesture samples")
    parser.add_argument("--train", action="store_true", help="Train from samples")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    parser.add_argument("--samples", default="assets/samples.npz", help="Path to samples file")
    parser.add_argument("--output", default=GESTURE_MODEL_PATH, help="Output model path")
    args = parser.parse_args()

    if args.collect:
        collect(args.samples)
    elif args.train:
        train(args.samples, args.output)
    elif args.verify:
        verify(args.output)
    else:
        parser.print_help()
