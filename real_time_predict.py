import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import math
import os
from gtts import gTTS
from pydub import AudioSegment
import pygame

# === Load model and label encoder ===
model = joblib.load('gesture_classifier_with_meta.joblib')
label_encoder = joblib.load('label_encoder_with_meta.joblib')

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Buffers and settings ===
SEQUENCE_LENGTH = 15
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=10)
CONFIDENCE_THRESHOLD = 0.7

spoken_label = None
AUDIO_FOLDER = "audio_wav"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# === Pygame sound setup ===
pygame.mixer.init()

# === Speak and cache audio ===
def speak_label(label):
    global spoken_label
    if label == spoken_label:
        return  # Don't repeat the same label

    spoken_label = label
    text = label.replace("_", " ")
    mp3_path = os.path.join(AUDIO_FOLDER, f"{label}.mp3")
    wav_path = os.path.join(AUDIO_FOLDER, f"{label}.wav")

    if not os.path.exists(wav_path):
        # Generate MP3 from text
        tts = gTTS(text)
        tts.save(mp3_path)

        # Convert MP3 to WAV
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        os.remove(mp3_path)  # Cleanup

    try:
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing audio: {e}")

# === Helper: angle between wrist and index base ===
def calculate_angle(wrist, index_base):
    dx = index_base[0] - wrist[0]
    dy = index_base[1] - wrist[1]
    return math.degrees(math.atan2(dy, dx))

# === Start webcam ===
cap = cv2.VideoCapture(0)
print("🟢 PALM started. Show a gesture...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hands detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('PALM Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    hand_landmark_frames = []
    angle_sum = 0
    hand_count = len(results.multi_hand_landmarks)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
        hand_landmark_frames.append(landmarks)

        wrist = hand_landmarks.landmark[0]
        index_base = hand_landmarks.landmark[5]
        angle_sum += calculate_angle((wrist.x, wrist.y), (index_base.x, index_base.y))

    while len(hand_landmark_frames) < 2:
        hand_landmark_frames.append([0.0] * 63)

    combined = hand_landmark_frames[0] + hand_landmark_frames[1]
    sequence_buffer.append(combined)

    if len(sequence_buffer) == SEQUENCE_LENGTH:
        avg_angle = angle_sum / hand_count if hand_count > 0 else 0
        flat_sequence = np.array(sequence_buffer).flatten()
        feature_vector = np.append(flat_sequence, [hand_count, avg_angle])

        proba = model.predict_proba([feature_vector])[0]
        max_index = np.argmax(proba)
        confidence = proba[max_index]

        if confidence >= CONFIDENCE_THRESHOLD:
            label = label_encoder.inverse_transform([max_index])[0]
            prediction_buffer.append(label)
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)

            # Speak new label
            speak_label(most_common)

            cv2.putText(frame, f"{most_common} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, f"Low confidence ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    cv2.imshow('PALM Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
