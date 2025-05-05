import cv2
import mediapipe as mp
import numpy as np
import torch
import json
from collections import deque
import time # Added for potential FPS calculation or timing


# --- Import necessary components from your files ---
try:
    from dataloader import normalize_landmarks
    from models import (SignGRUClassifierAttention)

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure 'dataloader.py' and 'gru_attention.py' are in the same directory.")
    exit()

# --- Constants and Configuration ---
MODEL_INFO_PATH = 'experiments/gru_attention_info.json' # Path to your model info JSON
MODEL_WEIGHTS_PATH = 'experiments/gru_attention_best.pth' # <<<=== UPDATE THIS if your .pth file has a different name/path
SEQUENCE_LENGTH = 48  # Number of frames to collect for prediction (adjust as needed)
PREDICTION_THRESHOLD = 0.1 # Minimum confidence score to display prediction

# --- Load Model Information ---
try:
    with open(MODEL_INFO_PATH, 'r') as f:
        model_info = json.load(f)
    INPUT_SIZE = model_info['input_size']
    HIDDEN_SIZE = model_info['hidden_size']
    NUM_LAYERS = model_info['num_layers']
    DROPOUT = model_info['dropout']
    BIDIRECTIONAL = model_info['bidirectional']
    NUM_CLASSES = len(model_info['label_to_index'])
    # Convert string keys from JSON back to integers for direct indexing
    INDEX_TO_LABEL = {int(k): v for k, v in model_info['index_to_label'].items()}

except FileNotFoundError:
    print(f"Error: Model info file not found at {MODEL_INFO_PATH}")
    exit()
except KeyError as e:
    print(f"Error: Missing key in model info file: {e}")
    exit()
except Exception as e:
    print(f"An error occurred loading model info: {e}")
    exit()


# --- Landmark Processing Function (uses imported normalize_landmarks) ---
def extract_features(multi_hand_landmarks, multi_handedness, image_width, image_height):
    """
    Extracts the 132 features (norm_left, norm_right, wrist_left, wrist_right)
    using MediaPipe landmarks and handedness.
    """
    # Initialize features with zeros
    left_hand_landmarks_flat = np.zeros(63, dtype=np.float32)
    right_hand_landmarks_flat = np.zeros(63, dtype=np.float32)
    left_wrist_raw = np.zeros(3, dtype=np.float32)
    right_wrist_raw = np.zeros(3, dtype=np.float32)
    found_left = False
    found_right = False

    if multi_hand_landmarks and multi_handedness:
        for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
            handedness_classification = multi_handedness[hand_idx].classification[0]
            hand_label = handedness_classification.label # Typically 'Left' or 'Right'

            # Extract landmarks into a flat list (x, y, z for each of 21 landmarks)
            temp_landmarks = []
            for lm in hand_landmarks.landmark:
                 temp_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_flat = np.array(temp_landmarks, dtype=np.float32)

            # Extract raw wrist coordinates (relative to image size)
            raw_wrist = np.array([
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x * image_width,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y * image_height,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z # Z might not need scaling
            ], dtype=np.float32)

            # Assign to left or right based on MediaPipe classification
            if hand_label == 'Left':
                left_hand_landmarks_flat = landmarks_flat
                left_wrist_raw = raw_wrist
                found_left = True
            elif hand_label == 'Right':
                right_hand_landmarks_flat = landmarks_flat
                right_wrist_raw = raw_wrist
                found_right = True

    # Normalize landmarks relative to their respective wrists using the imported function
    left_norm = normalize_landmarks(left_hand_landmarks_flat if found_left else np.zeros(63))
    right_norm = normalize_landmarks(right_hand_landmarks_flat if found_right else np.zeros(63))

    # Concatenate features: norm_left (63), norm_right (63), raw_wrist_left (3), raw_wrist_right (3)
    features = np.concatenate([left_norm, right_norm, left_wrist_raw, right_wrist_raw]) # Total 132 features
    return features


# --- Initialize Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the imported model class
model = SignGRUClassifierAttention(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
).to(device)

# Load weights
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"Model weights loaded successfully from {MODEL_WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_WEIGHTS_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0, # Use 0 for lighter model, 1 for heavier
    max_num_hands=2,
    min_detection_confidence=0.6, # Increased slightly
    min_tracking_confidence=0.6) # Increased slightly

# --- Initialize OpenCV Video Capture ---
cap = cv2.VideoCapture(0) # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Real-time Processing Loop ---
feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "Initializing..."
prediction_confidence = 0.0
last_print_time = time.time()

print("Starting live demo... Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB before processing.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # To improve performance, optionally mark the image as not writeable.
    rgb_image.flags.writeable = False
    results = hands.process(rgb_image)
    rgb_image.flags.writeable = True # No longer needed in RGB after processing

    # Extract features - Now includes handedness from results
    features = extract_features(results.multi_hand_landmarks, results.multi_handedness, image_width, image_height)
    feature_sequence.append(features)

    # Draw landmarks on the original BGR image
    display_image = image # Work on the original BGR image for display
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # Points
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2) # Connections
            )

    # Perform prediction when sequence is full
    if len(feature_sequence) == SEQUENCE_LENGTH:
        sequence_np = np.array(feature_sequence) # Shape: (SEQUENCE_LENGTH, INPUT_SIZE)
        # Ensure the sequence doesn't contain NaNs or Infs (can happen with bad normalization)
        if np.any(np.isnan(sequence_np)) or np.any(np.isinf(sequence_np)):
             if time.time() - last_print_time > 2: # Avoid flooding console
                 print("Warning: Invalid values (NaN/Inf) detected in feature sequence. Skipping prediction.")
                 last_print_time = time.time()
             current_prediction = "Invalid Data"
             prediction_confidence = 0.0
        else:
            sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim -> (1, SEQ_LEN, INPUT_SIZE)

            with torch.no_grad(): # Ensure gradients aren't calculated
                outputs = model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                predicted_idx = predicted_idx.item()
                prediction_confidence = confidence.item()

                if prediction_confidence >= PREDICTION_THRESHOLD:
                    current_prediction = INDEX_TO_LABEL.get(predicted_idx, f"Unknown Idx:{predicted_idx}")
                else:
                    current_prediction = "..." # Indicate low confidence / no clear sign


    # Display prediction
    cv2.putText(display_image, f"Prediction: {current_prediction} ({prediction_confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display_image, f"Frames: {len(feature_sequence)}/{SEQUENCE_LENGTH}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)


    # Show the image
    cv2.imshow('Sign Language Demo (GRU+Attention)', display_image)

    # Exit on 'q' key press
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Optional: Add a key to clear the sequence manually
        feature_sequence.clear()
        current_prediction = "Cleared"
        prediction_confidence = 0.0
        print("Sequence cleared.")


# --- Cleanup ---
hands.close()
cap.release()
cv2.destroyAllWindows()

print("Demo finished.")