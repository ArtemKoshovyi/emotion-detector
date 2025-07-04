import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import pickle

# --- Configuration ---
data_dir = "landmark_data"
important_idxs = [33, 133, 362, 263, 61, 291, 1, 13, 14, 17, 84, 91, 146]

# --- Load data ---
data = []
labels = []
for file in os.listdir(data_dir):
    if not file.endswith(".json"):
        continue
    with open(os.path.join(data_dir, file)) as f:
        obj = json.load(f)
        if "emotion" not in obj or "landmarks" not in obj:
            continue
        landmarks = obj["landmarks"]
        if len(landmarks) != len(important_idxs) * 3:
            continue
        emotion = obj["emotion"]
        data.append(landmarks)
        labels.append(emotion)

if len(data) < 2:
    print("Not enough samples to train. Check landmark_data/")
    exit()

X = np.array(data)
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

le = LabelEncoder()
y_raw = le.fit_transform(labels)
y_cat = to_categorical(y_raw)

# --- Stratified split to maintain class balance ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_raw
)

# --- Model architecture ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(X[0]),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

print("\n=== Test Accuracy ===")
loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc:.2f}")

print(classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1), target_names=le.classes_))

model.save("landmark_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model and LabelEncoder saved")
