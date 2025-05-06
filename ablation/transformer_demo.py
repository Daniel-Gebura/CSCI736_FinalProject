################################################################
# transformer_demo.py
#
# Description: Real-time ASL sign recognition demo using the
#              SignTransformerClassifier model. Captures webcam
#              input, extracts MediaPipe landmarks, preprocesses
#              them, feeds them to the trained Transformer model,
#              and displays the predicted sign.
#
# Usage: Run from the 'CSCI736_FinalProject/ablation/' directory.
#        Ensure 'models/transformer_classifier.py' and
#        'dataloader.py' exist and required packages are installed.
#
# Adapted from: GRU demo script
################################################################

import cv2
import mediapipe as mp
import numpy as np
import torch
import json
from collections import deque
import time
import math  # Needed for PositionalEncoding if used directly, but not for inference

# --- Import necessary components ---
try:
    # Assuming running from the 'ablation' directory
    from dataloader import normalize_landmarks  # For preprocessing
    from models.transformer_classifier import SignTransformerClassifier  # Import the Transformer model
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure:")
    print("1. You are running this script from the 'CSCI736_FinalProject/ablation/' directory.")
    print("2. 'dataloader.py' exists in the 'ablation' directory.")
    print("3. 'models/transformer_classifier.py' exists.")
    exit()

# --- Constants and Configuration ---
# Paths relative to the 'ablation' directory
MODEL_INFO_PATH = 'experiments/saved_models/transformer_info.json'
MODEL_WEIGHTS_PATH = 'experiments/saved_models/transformer_best.pth'
SEQUENCE_LENGTH = 48  # Number of frames for prediction (adjust based on training)
PREDICTION_THRESHOLD = 0.1  # Minimum confidence score to display prediction
DISPLAY_CONFIDENCE = True  # Set to False to hide confidence score in display

# --- Load Model Information ---
try:
    with open(MODEL_INFO_PATH, 'r') as f:
        model_info = json.load(f)
    # --- Parameters specific to SignTransformerClassifier ---
    INPUT_SIZE = model_info['input_size']
    HIDDEN_SIZE = model_info['hidden_size']  # Corresponds to d_model
    NUM_LAYERS = model_info['num_layers']  # Corresponds to num_encoder_layers
    NHEAD = model_info['nhead']
    DIM_FEEDFORWARD = model_info['dim_feedforward']
    DROPOUT = model_info['dropout']
    NUM_CLASSES = len(model_info['label_to_index'])
    # ---------------------------------------------------------
    # Convert string keys from JSON back to integers for direct indexing
    INDEX_TO_LABEL = {int(k): v for k, v in model_info['index_to_label'].items()}
    print("Model info loaded successfully.")
    print(f"Input Size: {INPUT_SIZE}, Hidden Size (d_model): {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Heads: {NHEAD}")

except FileNotFoundError:
    print(f"Error: Model info file not found at {MODEL_INFO_PATH}")
    exit()
except KeyError as e:
    print(f"Error: Missing key in model info file: {e}. Needed for Transformer.")
    exit()
except Exception as e:
    print(f"An error occurred loading model info: {e}")
    exit()


# --- Landmark Processing Function (from GRU demo, should be compatible) ---
def extract_features(multi_hand_landmarks, multi_handedness, image_width, image_height):
    """
    Extracts landmark features (normalized left/right hands + raw wrists).
    Output size should match model's INPUT_SIZE (e.g., 132).
    """
    # Initialize features with zeros (63 for each hand's norm landmarks + 3 for each wrist)
    left_hand_landmarks_flat = np.zeros(63, dtype=np.float32)
    right_hand_landmarks_flat = np.zeros(63, dtype=np.float32)
    left_wrist_raw = np.zeros(3, dtype=np.float32)
    right_wrist_raw = np.zeros(3, dtype=np.float32)
    found_left = False
    found_right = False

    if multi_hand_landmarks and multi_handedness:
        # Check if the number of landmarks and handedness info match
        if len(multi_hand_landmarks) != len(multi_handedness):
            print(
                f"Warning: Mismatch between detected hands ({len(multi_hand_landmarks)}) and handedness info ({len(multi_handedness)}). Skipping frame.")
            # Return zeros, which might lead to 'Invalid Data' prediction later
            return np.concatenate([np.zeros(63), np.zeros(63), np.zeros(3), np.zeros(3)])

        for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
            # Ensure handedness classification exists for the current hand index
            if hand_idx >= len(multi_handedness) or not multi_handedness[hand_idx].classification:
                print(f"Warning: Missing handedness classification for hand index {hand_idx}. Skipping this hand.")
                continue  # Skip this hand if handedness info is missing

            handedness_classification = multi_handedness[hand_idx].classification[0]
            hand_label = handedness_classification.label  # 'Left' or 'Right'

            # Extract landmarks into a flat list (x, y, z for each of 21 landmarks)
            temp_landmarks = []
            for lm in hand_landmarks.landmark:
                temp_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_flat = np.array(temp_landmarks, dtype=np.float32)  # Shape (63,)

            # Extract raw wrist coordinates (relative to image size)
            raw_wrist = np.array([
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x * image_width,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y * image_height,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z  # Z might not need scaling
            ], dtype=np.float32)  # Shape (3,)

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
    # Ensure the input to normalize_landmarks is always shape (63,)
    left_norm = normalize_landmarks(left_hand_landmarks_flat if found_left else np.zeros(63))
    right_norm = normalize_landmarks(right_hand_landmarks_flat if found_right else np.zeros(63))

    # Concatenate features: norm_left (63), norm_right (63), raw_wrist_left (3), raw_wrist_right (3)
    features = np.concatenate([left_norm, right_norm, left_wrist_raw, right_wrist_raw])  # Total 132 features

    # --- Crucial Check: Ensure feature size matches model input size ---
    if features.shape[0] != INPUT_SIZE:
        print(
            f"FATAL ERROR: Extracted feature size ({features.shape[0]}) does not match model INPUT_SIZE ({INPUT_SIZE}).")
        print("Check extract_features function and model info.")
        exit()
    # --------------------------------------------------------------------

    return features


# --- Initialize Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the SignTransformerClassifier model
model = SignTransformerClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,  # d_model
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    nhead=NHEAD,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

# Load weights
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Transformer model weights loaded successfully from {MODEL_WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_WEIGHTS_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    # Common issue: Mismatch between saved state_dict keys and current model definition
    print("Check if the 'transformer_classifier.py' definition matches the trained model.")
    exit()

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,  # 0: Lite, 1: Full
    max_num_hands=2,
    min_detection_confidence=0.6,  # Adjust as needed
    min_tracking_confidence=0.6)  # Adjust as needed

# --- Initialize OpenCV Video Capture ---
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Real-time Processing Loop ---
feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "Initializing..."
prediction_confidence = 0.0
last_print_time = time.time()
frame_count = 0  # For potential FPS calculation

print(f"Starting live demo with Transformer model...")
print(f"Collecting {SEQUENCE_LENGTH} frames for each prediction.")
print("Press 'q' to quit.")
print("Press 'c' to clear the current sequence.")

while cap.isOpened():
    start_time = time.time()  # Start time for FPS calculation

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_count += 1

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB before processing.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Process the image and find hands
    rgb_image.flags.writeable = False  # Improve performance
    results = hands.process(rgb_image)
    rgb_image.flags.writeable = True  # No longer needed in RGB

    # Extract features using the function defined above
    features = extract_features(results.multi_hand_landmarks, results.multi_handedness, image_width, image_height)
    feature_sequence.append(features)

    # Draw landmarks on the original BGR image for display
    display_image = image  # Work on the BGR image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  # Points style
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connections style
            )

    # Perform prediction only when the sequence buffer is full
    if len(feature_sequence) == SEQUENCE_LENGTH:
        sequence_np = np.array(feature_sequence)  # Shape: (SEQUENCE_LENGTH, INPUT_SIZE)

        # Check for invalid values before creating tensor
        if np.any(np.isnan(sequence_np)) or np.any(np.isinf(sequence_np)):
            if time.time() - last_print_time > 2:  # Avoid console flooding
                print("Warning: Invalid values (NaN/Inf) detected in feature sequence. Skipping prediction.")
                last_print_time = time.time()
            current_prediction = "Invalid Data"
            prediction_confidence = 0.0
        else:
            # Create tensor and add batch dimension -> (1, SEQ_LEN, INPUT_SIZE)
            sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model(sequence_tensor)  # Pass tensor to the Transformer model
                # Note: We are not passing 'lengths' here, so the model won't use padding masks.
                # This is okay if SEQUENCE_LENGTH is fixed and matches training,
                # or if the model handles sequences without masks robustly.

                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                predicted_idx_item = predicted_idx.item()
                prediction_confidence = confidence.item()

                # Update prediction text based on threshold
                if prediction_confidence >= PREDICTION_THRESHOLD:
                    current_prediction = INDEX_TO_LABEL.get(predicted_idx_item, f"Unknown Idx:{predicted_idx_item}")
                else:
                    current_prediction = "..."  # Indicate low confidence

    # --- Display Information on Frame ---
    # Prediction Text
    pred_text = f"Prediction: {current_prediction}"
    if DISPLAY_CONFIDENCE and current_prediction != "Initializing..." and current_prediction != "Invalid Data" and current_prediction != "...":
        pred_text += f" ({prediction_confidence:.2f})"
    cv2.putText(display_image, pred_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Sequence Buffer Status
    cv2.putText(display_image, f"Frames: {len(feature_sequence)}/{SEQUENCE_LENGTH}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    # Calculate and display FPS (optional)
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(display_image, f"FPS: {fps:.1f}",
                (image_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # --- Show the image ---
    cv2.imshow('ASL Sign Recognition - Transformer Demo', display_image)

    # --- Handle Key Presses ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear sequence buffer
        feature_sequence.clear()
        current_prediction = "Cleared"
        prediction_confidence = 0.0
        print("Sequence cleared.")

# --- Cleanup ---
print("Cleaning up...")
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Demo finished.")
