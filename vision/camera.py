"""
vision/camera.py — Webcam frame capture.

TODO (Nirmit): Port from old project.
"""

import cv2


class Camera:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(camera_index)
        # Ensure requested resolution is applied; some backends ignore set until opened
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera at index {camera_index} cannot be opened. Check camera connection.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # mirror for intuitive selfie view

    def release(self):
        self.cap.release()
