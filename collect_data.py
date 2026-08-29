import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
import json
import math

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Constants
DATA_DIR = "collected_data_npy"
GESTURES = ['thumbs_up', 'thumbs_down', 'fever', 'home', 'me', 'you', 'this', 'okay', 'angry', 'good_morning','hello','thank_you', 'welcome', 'wait', 'super', 'food','drink','photo','vision']
SEQUENCE_LENGTH = 15  # Number of frames per sample

# Create folders
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

print("Available gestures:", GESTURES)
gesture = input("Enter gesture name to record: ")

if gesture not in GESTURES:
    print(f"Invalid gesture. Please use one of: {GESTURES}")
    exit()

cap = cv2.VideoCapture(0)
sequence = []
sample_count = 0
max_samples = 30  # Number of full sequences

def calculate_angle(wrist, index_base):
    dx = index_base[0] - wrist[0]
    dy = index_base[1] - wrist[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

while cap.isOpened() and sample_count < max_samples:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = []
    hand_count = 0
    angle_sum = 0

    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            frame_landmarks.append(landmarks)

            # Calculate angle for this hand (wrist to index base)
            wrist = hand_landmarks.landmark[0]
            index_base = hand_landmarks.landmark[5]
            angle_sum += calculate_angle((wrist.x, wrist.y), (index_base.x, index_base.y))

    # Fill empty if no hand
    if not frame_landmarks:
        frame_landmarks.append([0.0] * 63)  # 21 landmarks * 3 coords

    # Pad for two hands if only one detected
    while len(frame_landmarks) < 2:
        frame_landmarks.append([0.0] * 63)

    # Flatten both hands into one frame feature
    combined = frame_landmarks[0] + frame_landmarks[1]
    sequence.append(combined)

    # Show preview
    cv2.putText(frame, f"{gesture} Seq: {len(sequence)}/{SEQUENCE_LENGTH}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting gesture sequence", frame)

    # Save sequence when full
    if len(sequence) == SEQUENCE_LENGTH:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        npy_path = os.path.join(DATA_DIR, gesture, f"{timestamp}.npy")
        np.save(npy_path, np.array(sequence))

        avg_angle = angle_sum / hand_count if hand_count > 0 else 0
        meta = {
            "hand_count": hand_count,
            "average_angle": avg_angle
        }
        json_path = npy_path.replace('.npy', '_meta.json')
        with open(json_path, 'w') as jf:
            json.dump(meta, jf, indent=2)

        print(f"Saved: {npy_path} with {json_path}")
        sequence = []
        sample_count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Sequence data collection complete.")
