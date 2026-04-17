"""
ThereSyn — AR Theremin-Synthesizer
Configuration constants
"""

# === Video / Camera ===
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# === Vision (MediaPipe) ===
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5

# === Gesture Detection ===
PINCH_THRESHOLD = 0.05          # normalized distance between thumb tip + index tip
FINGER_TIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# === Spatial Smoothing ===
LANDMARK_SMOOTH_ALPHA = 0.35    # exponential moving average factor (0=no change, 1=no smoothing)

# === Theremin Mode ===
THEREMIN_FREQ_MIN = 130.81      # C3 Hz
THEREMIN_FREQ_MAX = 1046.50     # C6 Hz
THEREMIN_VOL_MIN = 0.0
THEREMIN_VOL_MAX = 1.0
THEREMIN_X_DEADZONE = 0.05      # ignore tiny movements
THEREMIN_Y_DEADZONE = 0.05
THEREMIN_Y_INVERTED = True      # hand higher = louder (screen coords are inverted)

# === Note Generation ===
NOTE_DURATION = 0.5              # seconds (for pre-generated samples if needed)

# === System UI ===
SYSTEM_DEBOUNCE = 0.6           # seconds between system button triggers

# === DSP / Audio ===
SAMPLE_RATE = 44100
BIT_DEPTH = 16
CHANNELS = 2                    # stereo
BUFFER_SIZE = 512               # pygame mixer buffer
OSC_WAVEFORMS = ["sine", "sawtooth", "square", "triangle"]
DEFAULT_WAVEFORM = "sine"

# === Latency Profiling ===
LATENCY_LOG_ENABLED = True
LATENCY_LOG_INTERVAL = 2.0      # seconds between latency reports

# === ML Classifier ===
GESTURE_MODEL_PATH = "assets/gesture_classifier.onnx"
GESTURE_LABELS = ["piano_rest", "theremin_engage", "octave_up", "octave_down",
                  "instrument_switch", "play", "stop"]
GESTURE_CONFIDENCE_THRESHOLD = 0.6

# === MIDI ===
MIDI_ENABLED = True
MIDI_OUTPUT_NAME = "ThereSyn"
MIDI_PITCH_BEND_RANGE_SEMITONES = 2  # pitch bend range in semitones (±)

# === UI ===
UI_FONT_NAME = "Segoe UI"
UI_FONT_MEDIUM_SIZE = 16
UI_FONT_SMALL_SIZE = 12
UI_FONT_LARGE_SIZE = 22
