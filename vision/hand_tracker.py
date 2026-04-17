"""
vision/hand_tracker.py — MediaPipe hand landmark extraction.

TODO (Nirmit): Port from old project, same as before.
"""

import cv2
import mediapipe as mp
from config import MAX_HANDS, HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        )

    def process_frame(self, frame):
        """
        Extract hand landmarks from BGR frame.

        Returns:
            List of hand dicts: [{"type": "Left"|"Right", "landmarks": [(x,y,z)...], "raw_landmarks": mp_obj}, ...]
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_type = handedness.classification[0].label
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hands_data.append({
                    "type": hand_type,
                    "landmarks": landmarks,
                    "raw_landmarks": hand_landmarks,
                })

        return hands_data
