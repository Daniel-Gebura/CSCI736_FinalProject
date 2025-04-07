import os
import zipfile
import numpy as np
from collections import Counter
from tqdm.auto import tqdm # Use auto for notebook/console compatibility
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence # For handling variable length sequences
from sklearn.model_selection import train_test_split # For stratified splitting
import matplotlib.pyplot as plt # For plotting training curves
import json # To save model metadata

# --- Data Loading Function (from previous step) ---
def load_landmark_data_from_zip(zip_path, landmarks_folder_in_zip="landmarks/"):
    """
    Loads landmark data and labels from .npy files within a ZIP archive.
    Handles both (frames, 2, 21, 3) and (frames, 63) shapes, reshaping/padding
    to a consistent (frames, 126) shape.

    Args:
        zip_path (str): Path to the .zip file containing the landmark data.
        landmarks_folder_in_zip (str): The name of the folder inside the ZIP
                                       archive where .npy files are located.
                                       Must end with '/'.

    Returns:
        tuple: A tuple containing:
            - all_landmarks (list): A list of numpy arrays, where each array
                                    represents the landmarks for one clip
                                    (shape: [num_frames, 126]).
            - all_labels (list): A list of integer labels corresponding to each
                                 element in all_landmarks.
            - label_to_index (dict): A dictionary mapping label strings to integers.
            - index_to_label (dict): A dictionary mapping integers back to label strings.
            - filenames (list): A list of the filenames processed, in order.
    """
    print(f"Loading data from: {zip_path}")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at: {zip_path}")

    all_landmarks = []
    label_strings = []
    filenames = []
    unique_labels = set()
    processed_file_count = 0 # Keep track of files attempted

    # --- First pass: Extract filenames and unique labels ---
    print("Scanning ZIP file for labels...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            npy_files = [
                f for f in file_list
                if f.startswith(landmarks_folder_in_zip) and f.lower().endswith('.npy')
            ]

            if not npy_files:
                raise ValueError(f"No .npy files found in '{landmarks_folder_in_zip}' within {zip_path}")

            print(f"Found {len(npy_files)} potential .npy files.")

            for filename in tqdm(npy_files, desc="Scanning files"):
                processed_file_count += 1
                # Normalize path separators just in case
                normalized_filename = filename.replace('\\', '/')
                base_filename = normalized_filename.split('/')[-1]
                try:
                    # Expected format: SomePrefix_ParticipantID_Label_ClipID.npy
                    parts = base_filename.split('_')
                    if len(parts) < 4:
                        print(f"Warning: Skipping unexpected filename format: {filename}")
                        continue
                    label = parts[2] # Assuming label is the third part
                    unique_labels.add(label)
                    label_strings.append(label)
                    filenames.append(filename) # Store original path from zip
                except Exception as e:
                    print(f"Warning: Error parsing filename '{filename}': {e}. Skipping.")
                    continue

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Error: '{zip_path}' is not a valid ZIP file or is corrupted.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during the initial scan: {e}")

    if not unique_labels:
        raise ValueError("No valid labels could be extracted from filenames.")

    print(f"Found {len(unique_labels)} unique labels from {len(filenames)} parseable files (out of {processed_file_count} potential files).")

    # --- Create label mappings ---
    sorted_labels = sorted(list(unique_labels))
    label_to_index = {label: i for i, label in enumerate(sorted_labels)}
    index_to_label = {i: label for i, label in enumerate(sorted_labels)}
    print("Created label mappings.")

    # --- Second pass: Load actual landmark data ---
    print("Loading landmark data and reshaping/padding...")
    all_labels_indexed = []
    loaded_filenames_in_order = []
    skipped_files = 0

    TARGET_FEATURE_DIM = 126 # 2 hands * 21 landmarks * 3 coords

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Use the 'filenames' list gathered in the first pass which contains valid paths
            for filename in tqdm(filenames, desc="Loading npy data"):
                # Re-parse the label from the filename to ensure alignment
                normalized_filename = filename.replace('\\', '/')
                base_filename = normalized_filename.split('/')[-1]
                try:
                    parts = base_filename.split('_')
                    # Re-check format robustness
                    if len(parts) < 4:
                        # This should ideally not happen if filenames list is correct
                        print(f"Internal Warning: Filename '{filename}' in loading list has unexpected format. Skipping.")
                        skipped_files += 1
                        continue

                    label_str = parts[2]
                    if label_str not in label_to_index:
                        print(f"Warning: Label '{label_str}' from filename '{filename}' not found in mapping during loading phase. Skipping.")
                        skipped_files += 1
                        continue
                    label_idx = label_to_index[label_str]

                    with zip_ref.open(filename) as npy_file:
                        # Use io.BytesIO to load from the file-like object in memory
                        content = io.BytesIO(npy_file.read())
                        landmarks = np.load(content)
                        processed_landmarks = None
                        original_shape = landmarks.shape

                        # Case 1: Data includes two hands (frames, 2, 21, 3)
                        if len(landmarks.shape) == 4 and landmarks.shape[1:] == (2, 21, 3):
                            num_frames = landmarks.shape[0]
                            processed_landmarks = landmarks.reshape(num_frames, -1)
                            if processed_landmarks.shape[1] != TARGET_FEATURE_DIM:
                                print(f"Warning: Reshape failed for {filename}. Got shape {processed_landmarks.shape}. Skipping.")
                                skipped_files += 1
                                continue

                        # Case 2: Data includes only one hand (frames, 63)
                        elif len(landmarks.shape) == 2 and landmarks.shape[1] == 63:
                            num_frames = landmarks.shape[0]
                            zeros_padding = np.zeros((num_frames, 63), dtype=landmarks.dtype)
                            processed_landmarks = np.concatenate((landmarks, zeros_padding), axis=1)
                            if processed_landmarks.shape[1] != TARGET_FEATURE_DIM:
                                print(f"Warning: Padding failed for {filename}. Got shape {processed_landmarks.shape}. Skipping.")
                                skipped_files += 1
                                continue

                        # Case 3: Already in target shape (frames, 126) - less likely based on description but good to handle
                        elif len(landmarks.shape) == 2 and landmarks.shape[1] == TARGET_FEATURE_DIM:
                             processed_landmarks = landmarks

                        # Case 4: Unexpected shape
                        else:
                            print(f"Warning: Unexpected shape {original_shape} for {filename}. Expected compatible shape. Skipping.")
                            skipped_files += 1
                            continue

                        # Append successfully processed data
                        all_landmarks.append(processed_landmarks)
                        all_labels_indexed.append(label_idx)
                        loaded_filenames_in_order.append(filename)

                except KeyError: # Should be caught by the check above, but as safety
                    print(f"Internal Warning: Label '{label_str}' from filename '{filename}' caused KeyError. Skipping.")
                    skipped_files += 1
                except Exception as e:
                    print(f"Warning: Error loading or processing file '{filename}': {e}. Skipping.")
                    skipped_files += 1
                    continue

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Error: '{zip_path}' is not a valid ZIP file or is corrupted during data loading.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during data loading: {e}")

    print("-" * 40)
    print("Data Loading Summary:")
    print(f"Scanned {processed_file_count} potential .npy files.")
    print(f"Identified {len(filenames)} files with parseable labels.")
    print(f"Successfully loaded and processed clips: {len(all_landmarks)}")
    print(f"Skipped files during loading/processing: {skipped_files}")
    print(f"Number of unique labels mapped: {len(label_to_index)}")
    if all_landmarks:
        print(f"Feature dimension per frame: {all_landmarks[0].shape[1]}") # Should be 126

    # --- Verification ---
    if all_labels_indexed:
        label_counts = Counter(all_labels_indexed)
        print(f"\nLabel distribution (Top 5 loaded):")
        for label_idx, count in label_counts.most_common(5):
             if label_idx in index_to_label: # Check if index exists
                print(f"  {index_to_label[label_idx]}: {count}")
             else:
                print(f"  Index {label_idx} (Label missing?): {count}")


        print(f"\nLabel distribution (Bottom 5 loaded):")
        num_items_to_show = min(5, len(label_counts))
        # Sort by count ascending, then label string ascending for tie-breaking
        sorted_items = sorted(
            label_counts.items(),
            key=lambda item: (item[1], index_to_label.get(item[0], str(item[0])))
        )
        for label_idx, count in sorted_items[:num_items_to_show]:
            if label_idx in index_to_label: # Check if index exists
                print(f"  {index_to_label[label_idx]}: {count}")
            else:
                print(f"  Index {label_idx} (Label missing?): {count}")

        # Compare counts - adjust expected total based on successfully parsed filenames
        expected_successful_load_potential = len(filenames)
        if len(all_landmarks) != expected_successful_load_potential - skipped_files:
            print(f"\nWarning: Mismatch in counts. Loaded {len(all_landmarks)} clips.")
            print(f"         Expected based on parsed filenames minus skipped: {expected_successful_load_potential - skipped_files}")
            print(f"         Total parseable filenames initially: {expected_successful_load_potential}")
            print(f"         Files skipped during loading: {skipped_files}")

        # Base check for AUTSL might be 100 or 226 depending on subset
        # print(f"\nInfo: Expected unique labels based on AUTSL dataset can vary (e.g., 100 or 226). Found {len(label_to_index)}.")

    else:
        print("\nWarning: No data was successfully loaded after processing.")

    return all_landmarks, all_labels_indexed, label_to_index, index_to_label, loaded_filenames_in_order


# --- PyTorch Dataset ---
class SignLandmarkDataset(Dataset):
    def __init__(self, landmarks_list, labels_list):
        self.landmarks = [torch.tensor(seq, dtype=torch.float32) for seq in landmarks_list]
        self.labels = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

# --- Collate Function for Padding ---
def collate_fn(batch):
    """
    Pads sequences within a batch to the maximum sequence length in that batch.
    Args:
        batch: A list of tuples (sequence_tensor, label_tensor).
    Returns:
        A tuple (padded_sequences, labels, lengths).
        - padded_sequences: Tensor of shape (batch_size, max_seq_len, feature_dim).
        - labels: Tensor of shape (batch_size).
        - lengths: Tensor of shape (batch_size) holding original sequence lengths.
                   Useful for pack_padded_sequence if needed, or just for info.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # Pad sequences: batch_first=True makes shape (batch_size, max_len, features)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    # Stack labels: Ensure labels are returned as a tensor
    labels = torch.stack(labels, 0)
    return padded_sequences, labels, lengths


# --- GRU Model Definition ---
class SignGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=False) # Set bidirectional=True for potential improvement
        self.fc = nn.Linear(hidden_size, num_classes) # Use hidden_size * 2 if bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_size)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # Initial hidden state

        # Pack sequence is generally better if using variable lengths, but requires sorting
        # For simplicity here, we'll process the padded sequence directly.
        # The GRU handles padded sequences okay, but the final state needs care.
        gru_out, h_n = self.gru(x) # h_n shape: (num_layers * num_directions, batch_size, hidden_size)

        # We typically use the hidden state of the last time step for classification.
        # h_n contains the final hidden state for each element in the batch.
        # If bidirectional, h_n combines forward and backward states.
        # For unidirectional GRU, the last layer's final hidden state is h_n[-1]
        last_hidden_state = h_n[-1] # Shape: (batch_size, hidden_size)

        # Apply dropout and the final classification layer
        out = self.dropout(last_hidden_state)
        out = self.fc(out) # Shape: (batch_size, num_classes)
        return out


# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path, model_info_path):
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n--- Starting Training on {device} ---")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for sequences, labels, lengths in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths) # Pass lengths if needed by model variant
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0) # Accumulate loss weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for sequences, labels, lengths in val_pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': loss.item()})

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} => "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save the model if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved with Val Acc: {best_val_acc:.4f}")
            # Save model info alongside the best weights
            model_info = {
                'input_size': model.gru.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'num_classes': model.fc.out_features,
                'dropout': model.dropout.p,
                'bidirectional': model.gru.bidirectional,
                'label_to_index': label2idx, # Include the mappings
                'index_to_label': idx2label
            }
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            print(f"  -> Model info saved to {model_info_path}")


    print(f"\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model weights saved to: {model_save_path}")
    print(f"Model info saved to: {model_info_path}")

    return train_losses, val_losses, train_accs, val_accs


# --- Plotting Function ---
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png") # Save the plot
    print("Training curves plot saved as training_curves.png")
    # plt.show() # Optionally display the plot


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    ZIP_FILE_PATH = 'data/landmarks.zip'  # Adjust path if needed
    LANDMARKS_FOLDER_IN_ZIP = 'landmarks/' # Folder inside the zip
    MODEL_SAVE_DIR = 'saved_models' # Directory to save model files
    MODEL_NAME = 'sign_gru_classifier' # Base name for saved files
    MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_best.pth')
    MODEL_INFO_PATH = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_info.json')

    # Ensure save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # GRU Hyperparameters
    INPUT_SIZE = 126 # Determined by data loading (2 hands * 21 landmarks * 3 coords)
    HIDDEN_SIZE = 128 # Size of GRU hidden state (tune this)
    NUM_LAYERS = 2    # Number of stacked GRU layers (tune this)
    DROPOUT = 0.4     # Dropout rate (tune this)

    # Training Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50 # Adjust as needed based on convergence
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2 # 20% for validation

    # --- 1. Load Data ---
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Error: The file '{ZIP_FILE_PATH}' does not exist.")
        print("Please ensure the zip file is in the correct location.")
    else:
        try:
            all_landmarks, all_labels, label2idx, idx2label, fnames = load_landmark_data_from_zip(
                ZIP_FILE_PATH,
                LANDMARKS_FOLDER_IN_ZIP
            )

            if not all_landmarks:
                print("\nError: No data was loaded. Cannot proceed with training.")
            else:
                NUM_CLASSES = len(label2idx)
                print(f"\nData loaded successfully. Found {NUM_CLASSES} classes.")
                print(f"Total samples: {len(all_landmarks)}")

                # --- 2. Prepare Datasets and DataLoaders ---
                # Stratified split to maintain label distribution in train/val sets
                train_idx, val_idx = train_test_split(
                    list(range(len(all_labels))), # Indices
                    test_size=VALIDATION_SPLIT,
                    stratify=all_labels, # Use labels for stratification
                    random_state=42 # for reproducibility
                )

                train_landmarks = [all_landmarks[i] for i in train_idx]
                train_labels = [all_labels[i] for i in train_idx]
                val_landmarks = [all_landmarks[i] for i in val_idx]
                val_labels = [all_labels[i] for i in val_idx]

                train_dataset = SignLandmarkDataset(train_landmarks, train_labels)
                val_dataset = SignLandmarkDataset(val_landmarks, val_labels)

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) # No shuffle for validation

                print(f"Data split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
                print(f"Batch size: {BATCH_SIZE}")

                # --- 3. Initialize Model, Loss, Optimizer ---
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = SignGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, dropout=DROPOUT).to(device)
                criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                print("\nModel Summary:")
                print(model)
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total Trainable Parameters: {total_params:,}")

                # --- 4. Train the Model ---
                train_losses, val_losses, train_accs, val_accs = train_model(
                    model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device,
                    MODEL_WEIGHTS_PATH, MODEL_INFO_PATH
                )

                # --- 5. Plot Results ---
                plot_training_curves(train_losses, val_losses, train_accs, val_accs)

        except FileNotFoundError as e:
            print(e)
        except (ValueError, zipfile.BadZipFile, RuntimeError) as e:
            print(f"An error occurred during loading: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for unexpected errors
