import cv2
import os
import json
import mediapipe as mp
import numpy as np

# --- Settings ---
data_dir = "landmark_data"
os.makedirs(data_dir, exist_ok=True)
emotion_map = {
    ord('1'): "happy",
    ord('2'): "sad",
    ord('3'): "angry",
    ord('4'): "neutral"
}
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

video = cv2.VideoCapture(0)
print("Press 1–4 to label emotion (1-happy, 2-sad, 3-angry, 4-neutral), or 'q' to quit.")
counter = len([f for f in os.listdir(data_dir) if f.endswith(".json")])

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.multi_face_landmarks[0].landmark

        # Draw the face mesh
        mp.solutions.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=result.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

        # Wait for key press
        cv2.imshow("Collect Emotion Landmarks", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key in emotion_map:
            emotion = emotion_map[key]
            selected = []
            for i in important_idxs:
                lm = landmarks[i]
                selected.extend([lm.x, lm.y, lm.z])

            sample = {
                "emotion": emotion,
                "landmarks": [round(v, 6) for v in selected]
            }
            filename = os.path.join(data_dir, f"sample_{counter:04d}.json")
            with open(filename, 'w') as f:
                json.dump(sample, f)
            print(f"Saved {filename} as '{emotion}'")
            counter += 1
    else:
        cv2.imshow("Collect Emotion Landmarks", frame)

video.release()
cv2.destroyAllWindows()
