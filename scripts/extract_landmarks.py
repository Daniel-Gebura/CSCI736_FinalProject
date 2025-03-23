################################################################
# extract_landmarks.py
#
# Description: Batch extraction of MediaPipe hand landmarks from
# all ASL100 video clips. Output is saved as .npy arrays for training.
#
#
# Author: Daniel Gebura
################################################################

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Configuration Constants
CLIP_PATH = "../msasl/asl100_clips"     # Path to all sign video clips
OUTPUT_PATH = "../data/landmarks"       # Path to store .npy landmark files
NUM_HANDS = 2
NUM_LANDMARKS = 21

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize MediaPipe Hands API
mp_hands = mp.solutions.hands

def extract_landmarks_from_video(video_path, hands):
    """
    Extract (x, y, z) hand landmarks from a single video file.
    Returns a NumPy array of shape (frames, 2, 21, 3).

    Parameters:
        video_path (str): Path to the input video.
        hands (mediapipe.Hands): Initialized MediaPipe Hands object.

    Returns:
        np.ndarray: Array of shape (num_frames, 2, 21, 3)
                    where missing hands are zero-padded.
    """
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    sequence = []

    # 2. Process each frame in the video
    while cap.isOpened():
        # Read the next frame from the video
        success, frame = cap.read()
        if not success:
            break  # Reached end of video or error

        # Convert image from BGR to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Initialize default frame data (2 hands × 21 landmarks × 3 coords)
        frame_landmarks = np.zeros((NUM_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)

        # Extract landmarks if hands are detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Temporary dict to map hand type to its landmarks
            hand_dict = {
                "Left": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                "Right": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
            }

            # Loop through detected hands and their handedness info
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get label: "Left" or "Right"
                hand_label = handedness.classification[0].label

                # Extract each (x, y, z) landmark for this hand
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    hand_dict[hand_label][lm_idx] = [lm.x, lm.y, lm.z]

            # Reassign ordered hands to frame_landmarks: Left = [0], Right = [1]
            frame_landmarks[0] = hand_dict["Left"]
            frame_landmarks[1] = hand_dict["Right"]

        # Append landmarks for this frame to the full sequence
        sequence.append(frame_landmarks)

    cap.release()
    return np.array(sequence)

def main():
    video_files = [f for f in os.listdir(CLIP_PATH) if f.endswith(".mp4")]

    # Initialize MediaPipe Hands once and reuse
    with mp_hands.Hands(
        static_image_mode=False,  # Use detection+tracking mode
        max_num_hands=NUM_HANDS,  # Max number of hands to detect
        min_detection_confidence=0.5,  # Confidence threshold to detect a hand
        min_tracking_confidence=0.5  # Confidence threshold to continue tracking a hand
    ) as hands:

        # Process each video file ine the directory
        for filename in tqdm(video_files, desc="Extracting landmarks"):
            input_path = os.path.join(CLIP_PATH, filename)
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(OUTPUT_PATH, output_filename)

            if os.path.exists(output_path):
                continue  # Skip already-processed files

            landmarks = extract_landmarks_from_video(input_path, hands)

            if landmarks.shape[0] > 0:
                np.save(output_path, landmarks)

if __name__ == "__main__":
    main()