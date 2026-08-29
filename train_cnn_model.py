# train_cnn_gesture_model.py

import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# === Load Data ===
DATA_PATH = r'C:\Users\moham\OneDrive\Desktop\venv\venv\collected_data_npy'
gesture_folders = ['thumbs_up', 'thumbs_down', 'fever', 'home', 'me', 'you', 'this',
                   'okay', 'angry', 'good_morning', 'hello', 'thank_you', 'welcome',
                   'wait', 'super', 'food', 'drink', 'photo', 'vision']

X = []
y = []

for gesture in gesture_folders:
    folder_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(folder_path):
        print(f"Missing: {gesture}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            npy_path = os.path.join(folder_path, file)
            data = np.load(npy_path)
            if data.shape != (15, 126):
                continue
            X.append(data)
            y.append(gesture)

X = np.array(X)  # (N, 15, 126)
y = np.array(y)

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)
num_classes = y_onehot.shape[1]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y)

# === Build 1D CNN Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(15, 126)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train ===
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# === Save Model and Label Encoder ===
model.save("gesture_cnn_model.keras")
np.save("label_classes.npy", le.classes_)
print("Saved model as 'gesture_cnn_model.keras' and label encoder as 'label_classes.npy'")
