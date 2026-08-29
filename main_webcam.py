import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import math
import os
from gtts import gTTS
from pydub import AudioSegment
import pygame
import tensorflow as tf

# === Load CNN model and label encoder ===
model = tf.keras.models.load_model('gesture_cnn_model.keras')
label_classes = np.load('label_classes.npy')

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Buffers and settings ===
SEQUENCE_LENGTH = 15
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=10)
CONFIDENCE_THRESHOLD = 0.75
TEMPERATURE = 1.1
NUM_AUGMENTS = 2
NOISE_LEVEL = 0.01
MIN_STABLE_COUNT = 2

spoken_label = None
AUDIO_FOLDER = "audio_wav"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# === Sentence construction ===
sentence_buffer = deque(maxlen=2)
sentence_rules = {
    ("me", "fever"): "I have a fever",
    ("me", "food"): "I want food",
    ("me", "home"): "I want to go home",
    ("me", "angry"): "I am angry",
    ("me", "super"): "I am okay",
    ("me", "drink"): "I want water",
}
sentence_spoken = False

# === Pygame sound setup ===
pygame.mixer.init()

def speak_label(label):
    text = label.replace("_", " ")
    mp3_path = os.path.join(AUDIO_FOLDER, f"{label}.mp3")
    wav_path = os.path.join(AUDIO_FOLDER, f"{label}.wav")

    if not os.path.exists(wav_path):
        tts = gTTS(text)
        tts.save(mp3_path)
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        os.remove(mp3_path)

    try:
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")

# === Angle calculation ===
def calculate_angle(wrist, index_base):
    dx = index_base[0] - wrist[0]
    dy = index_base[1] - wrist[1]
    return math.degrees(math.atan2(dy, dx))

# === Softmax Temperature Scaling ===
def apply_temperature(predictions, temperature=1.5):
    logits = np.log(predictions + 1e-9)
    scaled_logits = logits / temperature
    exps = np.exp(scaled_logits)
    return exps / np.sum(exps)

# === Test-Time Augmentation (TTA) ===
def tta_predict(input_seq, model, num_augments=5, noise_level=0.01, temperature=1.5):
    preds = []
    for _ in range(num_augments):
        noisy = input_seq + np.random.normal(0, noise_level, input_seq.shape)
        raw_pred = model.predict(noisy, verbose=0)[0]
        smoothed = apply_temperature(raw_pred, temperature)
        preds.append(smoothed)
    return np.mean(preds, axis=0)

# === Stable Voting ===
def stable_prediction(buffer, min_count=3):
    if not buffer:
        return None
    most_common, count = Counter(buffer).most_common(1)[0]
    return most_common if count >= min_count else None

# === Start webcam ===
cap = cv2.VideoCapture(0)
print("\U0001F7E2 PALM (CNN version) started. Show a gesture...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hands detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('PALM Gesture Recognition (CNN)', frame)
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
        landmark_sequence = np.array(sequence_buffer)[:, :126]
        input_sequence = landmark_sequence.reshape(1, 15, 126)

        predictions = tta_predict(input_sequence, model, NUM_AUGMENTS, NOISE_LEVEL, TEMPERATURE)
        confidence = np.max(predictions)
        label_index = np.argmax(predictions)

        if confidence >= CONFIDENCE_THRESHOLD:
            label = label_classes[label_index]

            if label == "okay":
                label = "super"

            prediction_buffer.append(label)
            stable_label = stable_prediction(prediction_buffer, MIN_STABLE_COUNT)

            if stable_label:
                if not sentence_buffer or sentence_buffer[-1] != stable_label:
                    sentence_buffer.append(stable_label)
                    speak_label(stable_label)
                    sentence_spoken = False

                if len(sentence_buffer) == 2 and not sentence_spoken:
                    key = tuple(sentence_buffer)
                    sentence = sentence_rules.get(key)
                    if sentence:
                        speak_label(sentence)
                        sentence_spoken = True

                cv2.putText(frame, f"{stable_label} ({confidence*100:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f"Buffering... ({confidence*100:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"Low confidence ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    cv2.imshow('PALM Gesture Recognition (CNN)', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()