import os
import numpy as np
import json

DATA_PATH = r'C:\Users\moham\OneDrive\Desktop\venv\venv\collected_data_npy'

gesture_folders = ['thumbs_up', 'thumbs_down', 'fever', 'home', 'me', 'you', 'this', 'okay', 'angry', 'good_morning','hello','thank_you', 'welcome', 'wait', 'super', 'food','drink','photo','vision']

all_data = []
all_labels = []
all_meta = []

for gesture in gesture_folders:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        print(f"Warning: Folder not found {gesture_path}")
        continue

    for file in os.listdir(gesture_path):
        if file.endswith('.npy'):
            npy_path = os.path.join(gesture_path, file)
            data = np.load(npy_path)

            # Optional: Flatten if your model expects 1D input
            # data = data.flatten()  # Uncomment if needed

            all_data.append(data)
            all_labels.append(gesture)

            # Try loading corresponding metadata
            meta_path = npy_path.replace('.npy', '_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                meta = {"hand_count": None, "average_angle": None}
            all_meta.append(meta)

print(f"Loaded {len(all_data)} gesture sequences.")
print(f"Sample shape: {all_data[0].shape}")  # e.g., (15, 126)
print(f"Sample meta: {all_meta[0]}")
