import os
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Step 1: Load your data ===
DATA_PATH = r'C:\Users\moham\OneDrive\Desktop\venv\venv\collected_data_npy'

gesture_folders = ['thumbs_up', 'thumbs_down', 'fever', 'home', 'me', 'you', 'this', 'okay', 'angry', 'good_morning','hello','thank_you', 'welcome', 'wait', 'super', 'food','drink','photo','vision']

all_data = []
all_labels = []

print("Loading data with metadata...")

for gesture in gesture_folders:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        print(f"Warning: Folder not found {gesture_path}")
        continue

    for file in os.listdir(gesture_path):
        if file.endswith('.npy'):
            npy_path = os.path.join(gesture_path, file)
            json_path = npy_path.replace('.npy', '_meta.json')

            # Load landmark sequence and flatten
            data = np.load(npy_path)
            if data.shape != (15, 126):  # Ensure expected shape
                print(f"Skipping {npy_path}, shape mismatch: {data.shape}")
                continue

            flat_data = data.flatten()

            # Load metadata if available
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                hand_count = meta.get("hand_count", 0)
                average_angle = meta.get("average_angle", 0)
            else:
                hand_count = 0
                average_angle = 0

            # Combine landmarks + metadata
            combined = np.append(flat_data, [hand_count, average_angle])
            all_data.append(combined)
            all_labels.append(gesture)

print(f"Loaded {len(all_data)} samples with labels and metadata.")
print(f"Sample shape: {all_data[0].shape}")  # Should be (1892,)

# === Step 2: Preprocess data ===
X = np.array(all_data)
le = LabelEncoder()
y = le.fit_transform(all_labels)

print("Label classes:", le.classes_)

# === Step 3: Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# === Step 4: Train the classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training the Random Forest classifier...")
clf.fit(X_train, y_train)

# === Step 5: Evaluate the model ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Step 6: Save model and label encoder ===
model_filename = "gesture_classifier_with_meta.joblib"
label_encoder_filename = "label_encoder_with_meta.joblib"
joblib.dump(clf, model_filename)
joblib.dump(le, label_encoder_filename)
print(f"Saved model to '{model_filename}' and label encoder to '{label_encoder_filename}'")
