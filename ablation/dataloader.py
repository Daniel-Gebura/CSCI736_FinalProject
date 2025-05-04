################################################################
# dataloader.py
#
# Handles loading landmark data from a zip archive,
# preparing datasets, and providing dataloaders for training.
# Features per frame include:
#   - Normalized landmarks for each hand (relative to wrist)
#   - Raw wrist coordinates
# Padding duplicates the last valid frame.
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
    Processes each frame to include normalized landmarks and raw wrist coordinates.
    Padding duplicates last frame.

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

                        # Check for expected shape: (frames, 2, 21, 3)
                        if landmarks.ndim == 4 and landmarks.shape[1:] == (2, 21, 3):
                            num_frames = landmarks.shape[0]
                            feature_sequence = []

                            for frame_idx in range(num_frames):
                                frame = landmarks[frame_idx]

                                # Flatten both hands
                                left_hand = frame[0].flatten()
                                right_hand = frame[1].flatten()

                                # Normalize each hand
                                left_norm = normalize_landmarks(left_hand)
                                right_norm = normalize_landmarks(right_hand)

                                # Extract raw wrist positions
                                left_wrist = frame[0][0]  # (x, y, z)
                                right_wrist = frame[1][0]  # (x, y, z)

                                # Concatenate features
                                features = np.concatenate([
                                    left_norm,       # 63
                                    right_norm,      # 63
                                    left_wrist,      # 3
                                    right_wrist      # 3
                                ])  # → Final size: 132

                                feature_sequence.append(features)

                            feature_sequence = np.stack(feature_sequence)
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
    PyTorch Dataset for variable-length landmark sequences.
    """
    def __init__(self, landmarks_list, labels_list):
        # Convert each frame-sequence to a torch tensor of dtype float32
        self.landmarks = [torch.tensor(seq, dtype=torch.float32) for seq in landmarks_list]
        self.labels = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

# --- Collate Function ---
def collate_fn(batch):
    """
    Collate function that duplicates the last frame for padding.

    Args:
        batch (list): List of (sequence, label) pairs.

    Returns:
        tuple: (padded_sequences, labels, lengths)
            - padded_sequences: Tensor of shape (batch_size, max_len, feature_dim)
            - labels: Tensor of shape (batch_size)
            - lengths: Tensor of original sequence lengths
    """
    sequences, labels = zip(*batch)

    # Compute max length in batch
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    max_len = max(lengths)

    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            last_frame = seq[-1:]  # shape (1, feature_dim)
            padding = last_frame.repeat(pad_len, 1)  # repeat last frame
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    padded_sequences = torch.stack(padded_sequences, dim=0)  # shape (batch_size, max_len, feature_dim)
    labels = torch.stack(labels, dim=0)  # shape (batch_size)

    return padded_sequences, labels, lengths