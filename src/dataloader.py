################################################################
# dataloader.py
#
# Handles loading landmark data from a zip archive,
# preparing datasets, and providing dataloaders for training.
# Now includes normalized landmarks relative to wrist, raw wrist coords,
# and left/right hand marker.
#
# Author: Daniel Gebura
################################################################

import os
import zipfile
import numpy as np
import io
from collections import Counter
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- Helper Function: Normalize Landmarks Relative to Wrist ---
def normalize_landmarks(landmarks):
    """
    Normalize 21 hand landmarks relative to the wrist (landmark 0).

    Args:
        landmarks (list or np.ndarray): 63 elements (21 x 3D).

    Returns:
        np.ndarray: Flattened normalized landmarks (shape: (63,))
    """
    if len(landmarks) != 63:
        return np.zeros(63, dtype=np.float32)  # Handle invalid input safely
    landmarks = np.array(landmarks).reshape(21, 3)  # Reshape into (21, 3)
    wrist = landmarks[0]  # Wrist is landmark 0
    landmarks -= wrist  # Translate landmarks relative to wrist
    max_distance = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8  # Max distance for normalization
    return (landmarks / max_distance).flatten()  # Flatten back to (63,)

# --- Data Loading Function ---
def load_landmark_data_from_zip(zip_path, landmarks_folder_in_zip="../landmarks/"):
    """
    Loads landmark data and labels from a zipped folder.
    Processes each frame to add normalized landmarks, raw wrist 3D, and left/right markers.

    Args:
        zip_path (str): Path to .zip archive.
        landmarks_folder_in_zip (str): Folder inside ZIP with landmarks.

    Returns:
        tuple: (list of landmarks arrays, list of labels, label_to_index dict, index_to_label dict, list of filenames)
    """
    print(f"Loading data from: {zip_path}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    all_landmarks = []
    label_strings = []
    filenames = []
    unique_labels = set()
    processed_file_count = 0

    # --- First Pass: Identify Labels ---
    print("Scanning ZIP for labels...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            npy_files = [
                f for f in file_list
                if f.startswith(landmarks_folder_in_zip) and f.lower().endswith('.npy')
            ]

            if not npy_files:
                raise ValueError(f"No .npy files found inside {landmarks_folder_in_zip}")

            print(f"Found {len(npy_files)} .npy files.")

            for filename in tqdm(npy_files, desc="Scanning files"):
                processed_file_count += 1
                normalized_filename = filename.replace('\\', '/')
                base_filename = normalized_filename.split('/')[-1]
                try:
                    parts = base_filename.split('_')
                    if len(parts) < 4:
                        print(f"Skipping malformed filename: {filename}")
                        continue
                    label = parts[2]
                    unique_labels.add(label)
                    label_strings.append(label)
                    filenames.append(filename)
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Invalid ZIP file: {zip_path}")

    # --- Create Label Maps ---
    sorted_labels = sorted(list(unique_labels))
    label_to_index = {label: i for i, label in enumerate(sorted_labels)}
    index_to_label = {i: label for i, label in enumerate(sorted_labels)}
    print(f"Generated label mappings for {len(label_to_index)} classes.")

    # --- Second Pass: Load and Process Data ---
    all_labels_indexed = []
    loaded_filenames = []
    skipped_files = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in tqdm(filenames, desc="Loading landmarks"):
                normalized_filename = filename.replace('\\', '/')
                base_filename = normalized_filename.split('/')[-1]
                try:
                    parts = base_filename.split('_')
                    if len(parts) < 4:
                        skipped_files += 1
                        continue
                    label_str = parts[2]
                    if label_str not in label_to_index:
                        skipped_files += 1
                        continue
                    label_idx = label_to_index[label_str]

                    with zip_ref.open(filename) as npy_file:
                        content = io.BytesIO(npy_file.read())
                        landmarks = np.load(content)

                        if landmarks.ndim == 4 and landmarks.shape[1:] == (2, 21, 3):
                            num_frames = landmarks.shape[0]
                            feature_sequence = []

                            for frame_idx in range(num_frames):
                                frame = landmarks[frame_idx]

                                # Split into left and right hands
                                left_hand = frame[0].flatten()
                                right_hand = frame[1].flatten()

                                # Normalize landmarks
                                left_norm = normalize_landmarks(left_hand)
                                right_norm = normalize_landmarks(right_hand)

                                # Raw wrist coords
                                left_wrist = frame[0][0]  # (x, y, z) for left hand wrist
                                right_wrist = frame[1][0] # (x, y, z) for right hand wrist

                                # Marker (0 = left, 1 = right)
                                left_marker = np.array([0.0], dtype=np.float32)
                                right_marker = np.array([1.0], dtype=np.float32)

                                # Concatenate features into one long vector
                                features = np.concatenate([
                                    left_norm,       # Normalized left hand
                                    right_norm,      # Normalized right hand
                                    left_wrist,      # Raw left wrist
                                    right_wrist,     # Raw right wrist
                                    left_marker,     # Left marker
                                    right_marker     # Right marker
                                ])  # Final feature size: (63+63+3+3+1+1) = 132

                                feature_sequence.append(features)

                            feature_sequence = np.stack(feature_sequence)  # Stack frames
                            all_landmarks.append(feature_sequence)
                            all_labels_indexed.append(label_idx)
                            loaded_filenames.append(filename)

                        else:
                            skipped_files += 1
                            continue

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    skipped_files += 1

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Invalid ZIP file during second pass: {zip_path}")

    # --- Summary ---
    print(f"Loaded {len(all_landmarks)} clips, skipped {skipped_files} files.")

    return all_landmarks, all_labels_indexed, label_to_index, index_to_label, loaded_filenames

# --- Dataset Class ---
class SignLandmarkDataset(Dataset):
    """
    PyTorch Dataset for landmark sequences.
    """
    def __init__(self, landmarks_list, labels_list):
        self.landmarks = [torch.tensor(seq, dtype=torch.float32) for seq in landmarks_list]
        self.labels = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

# --- Collate Function ---
def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.

    Args:
        batch (list): List of (sequence, label) tuples.

    Returns:
        tuple: (padded_sequences, labels, lengths)
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, 0)
    return padded_sequences, labels, lengths
