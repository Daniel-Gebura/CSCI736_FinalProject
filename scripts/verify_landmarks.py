################################################################
# verify_landmarks.py
#
# Description: Randomly select 10 landmark files and display their
# corresponding videos with hand landmarks overlaid for verification.
#
# Author: Daniel Gebura
################################################################

import os
import random
import numpy as np
import cv2

# Configuration
LANDMARK_DIR = "../data/landmarks"       # Directory containing .npy landmark sequences
VIDEO_DIR = "../msasl/asl100_clips"      # Directory containing corresponding .mp4 video clips
NUM_HANDS = 2                             # Number of hands to expect (Left, Right)
NUM_LANDMARKS = 21                        # Number of landmarks per hand
NUM_SAMPLES = 10                          # Number of random samples to verify

def draw_landmarks_on_frame(frame, landmarks):
    """
    Draws hand landmarks on a single video frame.

    Parameters:
        frame (np.ndarray): The original frame image from the video.
        landmarks (np.ndarray): Array of shape (2, 21, 3), where:
            - 2 = number of hands (Left, Right)
            - 21 = number of landmarks per hand
            - 3 = x, y, z coordinates for each landmark (z ignored here)

    Returns:
        np.ndarray: Frame with green circles overlaid at each landmark position.
    """
    h, w, _ = frame.shape  # Get frame dimensions for coordinate scaling

    for hand_idx in range(NUM_HANDS):
        hand = landmarks[hand_idx]  # Get landmarks for this hand
        if np.sum(hand) > 0:  # Skip empty hands
            for lm in hand:
                # Convert normalized landmark coordinates to pixel coordinates
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                # Draw a small green circle at each landmark position
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    return frame  # Return frame with landmarks overlaid

def show_video_with_landmarks(video_path, landmark_path):
    """
    Plays a video with its corresponding landmarks overlaid frame-by-frame.

    Parameters:
        video_path (str): Path to the .mp4 video file.
        landmark_path (str): Path to the corresponding .npy landmark file.
    """
    landmark_sequence = np.load(landmark_path)  # Load the landmark sequence from disk
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frame_idx = 0  # Start from the first frame

    # Read and display each frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(landmark_sequence):
            break  # Stop if we reach the end of the video or landmark sequence

        # Get the landmarks for the current frame
        landmarks = landmark_sequence[frame_idx]

        # Draw the landmarks onto the frame
        frame = draw_landmarks_on_frame(frame, landmarks)

        # Display the result
        cv2.imshow("Landmark Overlay Verification", frame)

        # Allow early exit with 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1  # Move to next frame

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to randomly verify 10 landmark-video pairs.

    Steps:
    - Randomly choose 10 landmark files.
    - For each one, find and play the matching video.
    - Overlay landmarks for visual verification.
    """
    # Get all .npy landmark files
    landmark_files = [f for f in os.listdir(LANDMARK_DIR) if f.endswith(".npy")]
    if len(landmark_files) < NUM_SAMPLES:
        print(f"Not enough landmark files. Found {len(landmark_files)}.")
        return

    # Randomly select 10 files
    selected_files = random.sample(landmark_files, NUM_SAMPLES)

    # Process each selected file
    for i, selected in enumerate(selected_files, 1):
        print(f"\nPlaying sample {i}/{NUM_SAMPLES}: {selected}")

        # Build full paths to the landmark and video files
        landmark_path = os.path.join(LANDMARK_DIR, selected)
        video_filename = os.path.splitext(selected)[0] + ".mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)

        # Skip if the video file doesn't exist
        if not os.path.exists(video_path):
            print(f"Video not found: {video_filename}")
            continue

        # Show the video with landmarks
        show_video_with_landmarks(video_path, landmark_path)

    print("\nFinished verifying 10 random clips.")

# Entry point of the script
if __name__ == "__main__":
    main()