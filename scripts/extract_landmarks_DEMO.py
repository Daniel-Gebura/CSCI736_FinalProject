################################################################
# extract_landmarks_DEMO.py
#
# Description: This script extracts and visualizes hand landmarks 
# from a single video using MediaPipe Hands. It is designed to 
# demonstrate the preprocessing pipeline before scaling up to batch
# extraction.
#
# Author: Daniel Gebura
################################################################

import os
import cv2
import random
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Configuration Constants
DATASET_PATH = "../msasl/asl100_clips"  # Path to MS-ASL100 video folder
NUM_HANDS = 2  # Max number of hands to extract per frame
NUM_LANDMARKS = 21  # Number of landmarks per hand

# Initialize MediaPipe Hands API
mp_hands = mp.solutions.hands


def extract_landmarks_from_video(video_path):
    """
    Extracts (x, y, z) landmarks for up to 2 hands from each frame of a video.

    Parameters:
        video_path (str): Path to the input video.

    Returns:
        np.ndarray: Array of shape (num_frames, 2, 21, 3)
                    where missing hands are zero-padded.
    """
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    # 2. Initialize the MediaPipe Hands model
    hands = mp_hands.Hands(
        static_image_mode=False,  # Use detection+tracking mode
        max_num_hands=NUM_HANDS,  # Max number of hands to detect
        min_detection_confidence=0.5,  # Confidence threshold to detect a hand
        min_tracking_confidence=0.5  # Confidence threshold to continue tracking a hand
    )

    sequence = []  # Holds landmark data across all frames

    # 3. Process each frame in the video
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
            hand_dict = {"Left": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
                         "Right": np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)}

            # Loop through detected hands and their handedness info
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Get label: "Left" or "Right"
                hand_label = handedness.classification[0].label

                # Extract each (x, y, z) landmark for this hand
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    hand_dict[hand_label][lm_idx] = [lm.x, lm.y, lm.z]

            # Reassign ordered hands to frame_landmarks: Left → [0], Right → [1]
            frame_landmarks[0] = hand_dict["Left"]
            frame_landmarks[1] = hand_dict["Right"]

        # Append landmarks for this frame to the full sequence
        sequence.append(frame_landmarks)

    # 4. Release video capture and MediaPipe resources
    cap.release()
    hands.close()

    # 5. Convert list to NumPy array: shape (frames, 2, 21, 3)
    return np.array(sequence)


def plot_landmarks_sequence(sequence, video_path):
    """
    Visualizes all frames from the hand landmark sequence by overlaying them on the video frames.

    Parameters:
        sequence (np.ndarray): Array of shape (frames, 2, 21, 3)
        video_path (str): Path to the input video
    """
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frame_idx = 0  # Track which frame to get saved landmarks from

    # process each from in the video
    while cap.isOpened():
        # Get the next frame from the video
        success, frame = cap.read()
        if not success or frame_idx >= len(sequence):
            break  # End of video or sequence mismatch

        # Get the corresponding landmarks for this frame
        frame_landmarks = sequence[frame_idx]
        h, w, _ = frame.shape  # Get frame dimensions to scale landmarks (Stored [0.0, 1.0])

        # Draw landmarks for each hand if present
        for hand_idx in range(NUM_HANDS):
            hand = frame_landmarks[hand_idx]  # Get the landmarks for this hand
            if np.sum(hand) > 0:  # Only draw if landmarks exist
                # Loop through each landmark and draw it
                for lm in hand:
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)  # Draw green dot

        # Display the frame with overlays
        cv2.imshow("Hand Landmarks Overlay", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to:
    - Choose a random video from the dataset
    - Extract 2-hand landmarks per frame
    - Visualize a subset of the sequence
    """
    # Get list of all videos in the dataset path
    video_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".mp4") or f.endswith(".avi")]

    if not video_files:
        print(f"No video files found in: {DATASET_PATH}")
        return

    # Select a random video
    sample_filename = random.choice(video_files)
    sample_path = os.path.join(DATASET_PATH, sample_filename)
    print(f"Processing sample video: {sample_filename}")

    # Extract 2-hand landmark sequence from video
    sequence = extract_landmarks_from_video(sample_path)

    # Handle case where no landmarks were detected
    if len(sequence) == 0:
        print("No hand landmarks were detected in this video.")
        return

    print(f"Extracted {len(sequence)} frames.")
    print(f"Sequence shape: {sequence.shape}  (frames, 2 hands, 21 landmarks, 3 coords)")

    # Visualize the sequence
    plot_landmarks_sequence(sequence, sample_path)


if __name__ == "__main__":
    main()