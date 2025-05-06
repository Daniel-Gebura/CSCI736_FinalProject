################################################################
# demo.py
#
# Real-time gesture classification demo using a GRU+Attention model
# and MediaPipe Hands for landmark extraction.
#
# Author: Daniel Gebura, Yukti Gupta
################################################################

import cv2
import mediapipe as mp
import numpy as np
import torch
import json
from collections import deque
import time

# -------------------------------------------------------------
# Attempt to import project components (model + landmark utils)
# -------------------------------------------------------------
try:
    from dataloader import normalize_landmarks
    from model import SignGRUClassifierAttention
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'dataloader.py' and 'models.py' are available.")
    exit()

# ---------------------- Configuration ------------------------

MODEL_INFO_PATH = '../saved_models/sign_gru_classifier_info.json'       # Model metadata
MODEL_WEIGHTS_PATH = '../saved_models/sign_gru_classifier_best.pth'     # Trained weights
SEQUENCE_LENGTH = 48                                          # Frame history length
PREDICTION_THRESHOLD = 0.1                                    # Confidence threshold

# ---------------- Load Model Metadata ------------------------

try:
    with open(MODEL_INFO_PATH, 'r') as f:
        model_info = json.load(f)
    INPUT_SIZE     = model_info['input_size']
    HIDDEN_SIZE    = model_info['hidden_size']
    NUM_LAYERS     = model_info['num_layers']
    DROPOUT        = model_info['dropout']
    BIDIRECTIONAL  = model_info['bidirectional']
    NUM_CLASSES    = len(model_info['label_to_index'])
    INDEX_TO_LABEL = {int(k): v for k, v in model_info['index_to_label'].items()}
except Exception as e:
    print(f"Error loading model info: {e}")
    exit()

# ---------------- Feature Extraction -------------------------

def extract_features(multi_hand_landmarks, multi_handedness, image_width, image_height):
    """
    Extracts a consistent 132D feature vector for real-time demo that matches training format.

    Returns:
        np.ndarray: [norm_left(63), norm_right(63), raw_left_wrist(3), raw_right_wrist(3)]
    """
    # Default-zero features
    left_hand_flat = np.zeros(63, dtype=np.float32)
    right_hand_flat = np.zeros(63, dtype=np.float32)
    left_wrist = np.zeros(3, dtype=np.float32)
    right_wrist = np.zeros(3, dtype=np.float32)

    if multi_hand_landmarks and multi_handedness:
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            hand_label = multi_handedness[i].classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

            # Wrist position in image coordinates
            wrist = coords[0]
            wrist_pixel = np.array([
                wrist[0] * image_width,
                wrist[1] * image_height,
                wrist[2]
            ], dtype=np.float32)

            flat = coords.flatten()

            if hand_label == "Left":
                left_hand_flat = normalize_landmarks(flat)
                left_wrist = wrist_pixel
            elif hand_label == "Right":
                right_hand_flat = normalize_landmarks(flat)
                right_wrist = wrist_pixel

    return np.concatenate([left_hand_flat, right_hand_flat, left_wrist, right_wrist])

# -------------------- Model Initialization --------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SignGRUClassifierAttention(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
).to(device)

try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_WEIGHTS_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# -------------------- MediaPipe Setup ------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --------------------- Webcam Setup --------------------------

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# ------------------ Inference Loop ---------------------------

feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "Initializing..."
prediction_confidence = 0.0
last_warn_time = time.time()

print("Live demo running. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Skipped empty frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    features = extract_features(results.multi_hand_landmarks, results.multi_handedness, w, h)
    feature_sequence.append(features)

    # Draw hands
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # Predict if sequence full
    if len(feature_sequence) == SEQUENCE_LENGTH:
        seq = np.array(feature_sequence)
        if np.isnan(seq).any() or np.isinf(seq).any():
            if time.time() - last_warn_time > 2:
                print("Warning: NaN/Inf in features. Skipping.")
                last_warn_time = time.time()
            current_prediction = "Invalid"
            prediction_confidence = 0.0
        else:
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, dim=1)
                prediction_confidence = conf.item()
                current_prediction = INDEX_TO_LABEL.get(pred.item(), f"Unknown:{pred.item()}") if conf >= PREDICTION_THRESHOLD else "..."

    # Overlay prediction text
    cv2.putText(frame, f"Prediction: {current_prediction} ({prediction_confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Frames: {len(feature_sequence)}/{SEQUENCE_LENGTH}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Display video frame
    cv2.imshow("Sign Language Demo (GRU+Attention)", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        feature_sequence.clear()
        current_prediction = "Cleared"
        prediction_confidence = 0.0
        print("Sequence cleared.")

# ------------------------ Cleanup ----------------------------

hands.close()
cap.release()
cv2.destroyAllWindows()
print("Demo finished.")