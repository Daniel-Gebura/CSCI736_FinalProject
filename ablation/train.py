################################################################
# train.py
#
# Trains all model variants for gesture classification.
# Supports ablation studies by looping through:
#   - GRU baseline
#   - Bidirectional GRU
#   - Bidirectional GRU + LayerNorm
#   - Bidirectional GRU + MLP
#   - Bidirectional GRU + LayerNorm + MLP
#   - Bidirectional GRU + LayerNorm + MLP + Attention
#   - Transformer
#   - TCN
#
# Handles:
#   - Setting global seed
#   - Data loading
#   - Model instantiation
#   - Training and validation
#   - Logging best results and saving model state
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
from utils.experiment_logger import log_experiment_result
from models import (
    SignGRUClassifier,
    SignBiGRUClassifier,
    SignGRUClassifier_LayerNorm,
    SignGRUClassifier_MLP,
    SignGRUClassifier_LayerNorm_MLP,
    SignGRUClassifierAttention,
    # --- ADDED ---
    SignTransformerClassifier,
    SignTCNClassifier
    # -------------
)

# ---------------------------------------------------------------
# Main Training Execution Function
# ---------------------------------------------------------------
def train(model_class, model_name):
    """
    Trains a given model architecture using the configuration constants.

    Args:
        model_class (nn.Module): The class of the model to instantiate.
        model_name (str): A short name to identify this model variant (used for saving results).
    """
    # Set seed for reproducibility across numpy, torch, and Python RNGs
    set_seed(CONFIG['seed'])

    # Select training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create model save directory if it does not exist
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

    # Load landmark dataset from zipped .npy files
    all_landmarks, all_labels, label2idx, idx2label, _ = load_landmark_data_from_zip(
        CONFIG['zip_path'],
        CONFIG['landmarks_folder']
    )

    if not all_landmarks:
        print("Error: No data found to train on. Exiting.")
        return

    print(f"Total samples loaded: {len(all_landmarks)} | Classes: {len(label2idx)}")

    # Perform stratified train/validation split
    train_idx, val_idx = train_test_split(
        range(len(all_labels)),
        test_size=CONFIG['validation_split'],
        stratify=all_labels,
        random_state=CONFIG['seed']
    )

    # Split landmark data into training and validation sets
    train_data = [all_landmarks[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_data = [all_landmarks[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    # Construct PyTorch Dataset and DataLoaders
    train_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(train_data, train_labels),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(val_data, val_labels),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model with given architecture and move to device
    # Use a try-except block for potential init errors like hidden_size % nhead
    try:
        model = model_class(
            input_size=CONFIG['input_size'],
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            num_classes=len(label2idx),
            dropout=CONFIG['dropout']
        ).to(device)
    except Exception as e:
        print(f"\nError initializing model {model_name}: {e}")
        traceback.print_exc()
        return # Skip training this model variant

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Train model and track best validation accuracy
    try:
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
            # --- ADDED --- Pass early stopping patience from config
            early_stopping_patience=CONFIG['early_stopping_patience']
        )

        # Save metrics to results.csv
        log_experiment_result(model_name, best_val_acc, logs, CONFIG)

        # Plot training curves to PNG
        plot_training_curves(logs['train_losses'], logs['val_losses'], logs['train_accs'], logs['val_accs'], model_name)

    except Exception as e:
        print(f"\nError during training for model {model_name}: {e}")
        traceback.print_exc()

# ---------------------------------------------------------------
# Entry Point for All Ablation Runs
# ---------------------------------------------------------------
if __name__ == "__main__":
    """
    Entry point to train all model variants as part of an ablation study.
    Each model is trained, logged, and evaluated independently.
    """
    model_variants = [
        # ("gru_base", SignGRUClassifier),
        # ("gru_bidirectional", SignBiGRUClassifier),
        # ("gru_layernorm", SignGRUClassifier_LayerNorm),
        # ("gru_mlp", SignGRUClassifier_MLP),
        # ("gru_layernorm_mlp", SignGRUClassifier_LayerNorm_MLP),
        # ("gru_attention", SignGRUClassifierAttention),
        # --- ADDED ---
        ("transformer", SignTransformerClassifier),
        # ("tcn", SignTCNClassifier)
        # -------------
    ]

    # Ensure model save directory exists from config
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

    for model_name, model_class in model_variants:
        print(f"\n\n========== Running Model: {model_name} ==========")
        train(model_class, model_name)