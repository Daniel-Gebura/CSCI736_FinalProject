################################################################
# train.py
#
# Trains GRU and Transformer classifiers for gesture
# classification using shared training pipeline.
# Uses architecture-specific configuration from CONFIG.
#
# Author: Daniel Gebura
################################################################

import os
import json
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from config import CONFIG
from dataloader import load_landmark_data_from_zip, SignLandmarkDataset, collate_fn
from utils.metrics import set_seed, evaluate_model
from utils.visualization import plot_training_curves

# --- Models ---
from models.gru_classifier import SignGRUClassifierAttention

# ---------------------------------------------------------------
# Training Routine for a Given Model Variant
# ---------------------------------------------------------------
def train(model_class, model_name):
    """
    Trains a single model variant and logs results.

    Args:
        model_class (nn.Module): The model class to instantiate.
        model_name (str): Identifier for logs and saved files.
    """
    # Set global seed for reproducibility
    set_seed(CONFIG['seed'])

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create model save directory if it doesn't exist
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

    # Load landmark sequence data from zip
    all_landmarks, all_labels, label2idx, idx2label, _ = load_landmark_data_from_zip(
        CONFIG['zip_path'], CONFIG['landmarks_folder'])

    if not all_landmarks:
        print("Error: No data found.")
        return

    # Stratified train-validation split
    train_idx, val_idx = train_test_split(
        range(len(all_labels)),
        test_size=CONFIG['validation_split'],
        stratify=all_labels,
        random_state=CONFIG['seed']
    )

    # Subset train/val data
    train_data = [all_landmarks[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_data = [all_landmarks[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    # Create PyTorch dataloaders
    train_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(train_data, train_labels),
        batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(val_data, val_labels),
        batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

    try:
        # Base model arguments
        model_kwargs = {
            'input_size': CONFIG['input_size'],
            'num_classes': len(label2idx),
            'hidden_size': CONFIG['hidden_size'],
            'num_layers': CONFIG['num_layers'],
            'dropout': CONFIG['dropout']

        }

        # Instantiate model
        model = model_class(**model_kwargs).to(device)

    except Exception as e:
        print(f"Model initialization error: {e}")
        traceback.print_exc()
        return

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    try:
        # Train and evaluate
        best_val_acc, logs = evaluate_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=CONFIG['num_epochs'],
            device=device,
            save_dir=CONFIG['model_save_dir'],
            label2idx=label2idx,
            idx2label=idx2label,
            input_size_config=CONFIG['input_size'],
            early_stopping_patience=CONFIG['early_stopping_patience']
        )

        # Plot the training curves
        plot_training_curves(
            logs['train_losses'], logs['val_losses'],
            logs['train_accs'], logs['val_accs'], model_name)

    except Exception as e:
        print(f"Training error for {model_name}: {e}")
        traceback.print_exc()

# ---------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------
if __name__ == "__main__":
    model_variants = [
        ("gru_attention", SignGRUClassifierAttention),
    ]

    for model_name, model_class in model_variants:
        print(f"\n========== Training: {model_name} ==========")
        train(model_class, model_name)