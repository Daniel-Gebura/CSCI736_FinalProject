################################################################
# dataloader.py
#
# Handles loading landmark data from a zip archive,
# preparing datasets, and providing dataloaders for training.
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

# --- Data Loading Function ---
def load_landmark_data_from_zip(zip_path, landmarks_folder_in_zip="../landmarks/"):
    """
    Loads landmark data and labels from .npy files inside a zip archive.

    Args:
        zip_path (str): Path to the ZIP archive.
        landmarks_folder_in_zip (str): Subfolder inside the ZIP containing landmarks.

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

    # --- First pass: Scan file list to build label dictionary ---
    print("Scanning ZIP for labels...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            npy_files = [
                f for f in file_list
                if f.startswith(landmarks_folder_in_zip) and f.lower().endswith('.npy')
            ]

            if not npy_files:
                raise ValueError(f"No .npy files found in {landmarks_folder_in_zip}")

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

    # --- Create label mapping dictionaries ---
    sorted_labels = sorted(list(unique_labels))
    label_to_index = {label: i for i, label in enumerate(sorted_labels)}
    index_to_label = {i: label for i, label in enumerate(sorted_labels)}
    print(f"Generated label mappings for {len(label_to_index)} classes.")

    # --- Second pass: Load and process landmarks ---
    all_labels_indexed = []
    loaded_filenames = []
    skipped_files = 0
    TARGET_FEATURE_DIM = 126  # 2 hands × 21 landmarks × 3 coordinates

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
                        processed = None

                        # Handle different input formats
                        if len(landmarks.shape) == 4 and landmarks.shape[1:] == (2, 21, 3):
                            num_frames = landmarks.shape[0]
                            processed = landmarks.reshape(num_frames, -1)
                        elif len(landmarks.shape) == 2 and landmarks.shape[1] == 63:
                            num_frames = landmarks.shape[0]
                            zeros = np.zeros((num_frames, 63))
                            processed = np.concatenate((landmarks, zeros), axis=1)
                        elif len(landmarks.shape) == 2 and landmarks.shape[1] == 126:
                            processed = landmarks
                        else:
                            skipped_files += 1
                            continue

                        all_landmarks.append(processed)
                        all_labels_indexed.append(label_idx)
                        loaded_filenames.append(filename)

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
