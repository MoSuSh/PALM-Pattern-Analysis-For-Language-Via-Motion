# config.py

# Feature & Temporal Window Constants
SEQUENCE_LENGTH = 15      # Frames per gesture buffer
NUM_HANDS = 2             # Dual hand tracking
LANDMARKS_PER_HAND = 21   # MediaPipe standard joints
COORDS = 3                # X, Y, Z coordinates
FEATURE_DIM = NUM_HANDS * LANDMARKS_PER_HAND * COORDS  # 126 floats

# Model & Inference Settings
TEMPERATURE = 1.1         # Softmax temperature scaling
STABLE_WINDOW = 2         # Majority voting window length