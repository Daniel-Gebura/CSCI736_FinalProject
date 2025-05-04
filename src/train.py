################################################################
# train.py
#
# Handles the full training pipeline: 
# - Loads data
# - Loads model architecture
# - Trains model
# - Saves outputs
#
# Author: Daniel Gebura
################################################################

import os
import json
import random
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from dataloader import load_landmark_data_from_zip, SignLandmarkDataset, collate_fn
from model import SignGRUClassifierAttention

# ---------------- Configuration Constants ----------------

# Data Paths
ZIP_FILE_PATH = '../data/landmarks.zip'
LANDMARKS_FOLDER_IN_ZIP = 'landmarks/'
MODEL_SAVE_DIR = '../saved_models'
MODEL_NAME = 'sign_gru_classifier'

# Model Save Paths
MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_best.pth')
MODEL_INFO_PATH = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_info.json')

# Hyperparameters
INPUT_SIZE = 132            # Updated input size to match dataloader output (63+63+3+3+1+1)
HIDDEN_SIZE = 128           # GRU hidden size
NUM_LAYERS = 2              # GRU layers
DROPOUT = 0.5               # Dropout probability
BATCH_SIZE = 32             # Batch size
NUM_EPOCHS = 100            # Number of epochs
LEARNING_RATE = 0.001       # Learning rate
VALIDATION_SPLIT = 0.2      # Fraction of data for validation
WEIGHT_DECAY = 1e-5

# Seeding
SEED = 42                  # Set your seed value here for reproducibility

# -----------------------------------------------------------

# --- Set Random Seeds for Reproducibility ---
def set_seed(seed_value):
    """
    Sets random seeds across torch, numpy, and random to ensure reproducible results.

    Args:
        seed_value (int): The seed number.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path, model_info_path, label2idx, idx2label):
    """
    Trains a model using provided dataloaders.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        criterion (loss): Loss function.
        optimizer (optim): Optimizer.
        num_epochs (int): Number of training epochs.
        device (torch.device): Training device (CPU or GPU).
        model_save_path (str): Path to save best model weights.
        model_info_path (str): Path to save model metadata.
        label2idx (dict): Label to index mapping.
        idx2label (dict): Index to label mapping.

    Returns:
        tuple: (list of training losses, validation losses, training accuracies, validation accuracies)
    """
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
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
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

        print(f"Epoch {epoch+1}/{num_epochs} => Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # --- Save Best Model ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved with Val Acc: {best_val_acc:.4f}")
            model_info = {
                'input_size': model.gru.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.dropout.p,
                'bidirectional': model.gru.bidirectional,
                'label_to_index': label2idx,
                'index_to_label': idx2label
            }
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            print(f"  -> Model info saved to {model_info_path}")

    print(f"\n--- Training Complete ---\nBest Validation Accuracy: {best_val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs

# --- Plotting Function ---
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training and validation loss and accuracy curves.

    Args:
        train_losses (list): Training losses.
        val_losses (list): Validation losses.
        train_accs (list): Training accuracies.
        val_accs (list): Validation accuracies.
    """
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
    plt.savefig("training_curves.png")
    print("Training curves saved as training_curves.png")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure model save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Set random seed before anything else
    set_seed(SEED)

    # Load Dataset
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Error: File not found at {ZIP_FILE_PATH}")
    else:
        try:
            all_landmarks, all_labels, label2idx, idx2label, fnames = load_landmark_data_from_zip(ZIP_FILE_PATH, LANDMARKS_FOLDER_IN_ZIP)

            if not all_landmarks:
                print("No data loaded. Exiting.")
            else:
                NUM_CLASSES = len(label2idx)

                # Stratified train/validation split
                train_idx, val_idx = train_test_split(
                    list(range(len(all_labels))),
                    test_size=VALIDATION_SPLIT,
                    stratify=all_labels,
                    random_state=SEED
                )

                train_landmarks = [all_landmarks[i] for i in train_idx]
                train_labels = [all_labels[i] for i in train_idx]
                val_landmarks = [all_landmarks[i] for i in val_idx]
                val_labels = [all_labels[i] for i in val_idx]

                # Prepare datasets and dataloaders
                train_dataset = SignLandmarkDataset(train_landmarks, train_labels)
                val_dataset = SignLandmarkDataset(val_landmarks, val_labels)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

                # Device setup
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = SignGRUClassifierAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, dropout=DROPOUT).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

                # Train model
                train_losses, val_losses, train_accs, val_accs = train_model(
                    model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,
                    device, MODEL_WEIGHTS_PATH, MODEL_INFO_PATH, label2idx, idx2label
                )

                # Plot results
                plot_training_curves(train_losses, val_losses, train_accs, val_accs)

        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
