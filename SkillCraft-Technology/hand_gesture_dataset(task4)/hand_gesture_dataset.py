import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =============================
# Settings
# =============================
GESTURES = ["Fist", "Open", "Peace", "ThumbsUp"]
SAMPLES_PER_GESTURE = 40
MODEL_FILE = "gesture_model.pkl"
DATA_FILE = "gesture_data.pkl"

# =============================
# Initialize MediaPipe
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# =============================
# Function: Collect Dataset
# =============================
def collect_data():
    print("Dataset not found. Starting automatic data collection...")
    cap = cv2.VideoCapture(0)

    data = []
    labels = []

    for gesture in GESTURES:
        print(f"\nShow gesture: {gesture}")
        print("Press 'S' to save sample")

        count = 0
        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    landmark_list = []
                    for lm in handLms.landmark:
                        landmark_list.extend([lm.x, lm.y, lm.z])

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        data.append(landmark_list)
                        labels.append(gesture)
                        count += 1
                        print(f"{gesture} sample saved ({count}/{SAMPLES_PER_GESTURE})")

            cv2.putText(frame, f"Gesture: {gesture}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    with open(DATA_FILE, "wb") as f:
        pickle.dump((data, labels), f)

    print("Data collection completed!")

# =============================
# Function: Train Model
# =============================
def train_model():
    with open(DATA_FILE, "rb") as f:
        data, labels = pickle.load(f)

    X = np.array(data)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, le), f)

    print("Model trained and saved!")

# =============================
# Function: Real-time Prediction
# =============================
def predict():
    cap = cv2.VideoCapture(0)

    with open(MODEL_FILE, "rb") as f:
        model, le = pickle.load(f)

    print("Starting real-time recognition... Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture_text = "No Hand"

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in handLms.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                prediction = model.predict([landmark_list])
                gesture_text = le.inverse_transform(prediction)[0]

        cv2.putText(frame, f"Gesture: {gesture_text}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# MAIN PROGRAM
# =============================
if not os.path.exists(DATA_FILE):
    collect_data()
    train_model()
    predict()

elif not os.path.exists(MODEL_FILE):
    train_model()
    predict()

else:
    predict()