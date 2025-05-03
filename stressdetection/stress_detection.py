import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C + 1e-6)

def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[0] - mouth_points[6])
    B = np.linalg.norm(mouth_points[1] - mouth_points[7])
    C = np.linalg.norm(mouth_points[3] - mouth_points[9])
    return (A + B) / (2.0 * C + 1e-6)

def extract_features(landmarks, frame_shape):
    h, w = frame_shape[:2]
    features = []

    # Convert to pixel coordinates
    norm_landmarks = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    # 1. Eye Features
    left_eye = norm_landmarks[[33, 160, 158, 133, 153, 144]]
    right_eye = norm_landmarks[[362, 385, 387, 263, 373, 380]]
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    features.extend([left_ear, right_ear, np.abs(left_ear - right_ear)])

    # 2. Brow Features
    left_brow = np.mean(norm_landmarks[[70, 63, 105]][:,1])
    right_brow = np.mean(norm_landmarks[[336, 296, 334]][:,1])
    left_brow_tension = left_brow - norm_landmarks[160][1]
    right_brow_tension = right_brow - norm_landmarks[385][1]
    features.extend([left_brow_tension, right_brow_tension, np.abs(left_brow_tension - right_brow_tension)])

    # 3. Mouth Features
    mouth = norm_landmarks[[13, 14, 78, 80, 81, 82, 87, 88, 95, 178]]
    features.append(calculate_mar(mouth))

    # 4. Jaw Clenching
    features.append(np.linalg.norm(norm_landmarks[152] - norm_landmarks[4]))

    return np.array(features)

def load_model():
    try:
        model = joblib.load('enhanced_stress_model.pkl')
        if model.n_features_in_ != 8:
            print("Error: Model expects different features")
            sys.exit(1)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            try:
                landmarks = results.multi_face_landmarks[0].landmark
                features = extract_features(landmarks, frame.shape)

                stress_level = model.predict([features])[0]
                confidence = np.max(model.predict_proba([features]))

                # Display results
                colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
                cv2.putText(frame, f"{['Low','Medium','High'][stress_level]} Stress ({confidence:.0%})",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[stress_level], 2)

            except Exception as e:
                print(f"Processing error: {e}")
                continue

        cv2.imshow('Stress Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()