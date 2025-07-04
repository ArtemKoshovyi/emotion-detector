import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# --- Load model and LabelEncoder ---
model = load_model("landmark_model.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
labels = list(encoder.classes_)

# --- Important landmark indices ---
important_idxs = [33, 133, 362, 263, 61, 291, 1, 13, 14, 17, 84, 91, 146]

# --- Mediapipe configuration ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Open camera ---
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.multi_face_landmarks[0].landmark

        selected = []
        for i in important_idxs:
            lm = landmarks[i]
            selected.extend([lm.x, lm.y, lm.z])

        vec = np.array(selected)
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        vec = vec.reshape(1, -1)

        prediction = model.predict(vec, verbose=0)[0]
        emotion = labels[np.argmax(prediction)]

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=result.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

        cv2.putText(frame, f"Emotion: {emotion}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Light Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
