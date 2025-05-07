################################################################
# train.py
#
# Trains GRU, Transformer, and TCN classifiers for gesture
# classification using shared training pipeline.
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

# --- Models ---
from models.gru_classifier import SignGRUClassifierAttention
from models.transformer_classifier import SignTransformerClassifier
from models.tcn_classifier import SignTCNClassifier

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
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

    all_landmarks, all_labels, label2idx, idx2label, _ = load_landmark_data_from_zip(
        CONFIG['zip_path'], CONFIG['landmarks_folder'])

    if not all_landmarks:
        print("Error: No data found.")
        return

    # Stratified split
    train_idx, val_idx = train_test_split(
        range(len(all_labels)),
        test_size=CONFIG['validation_split'],
        stratify=all_labels,
        random_state=CONFIG['seed']
    )

    train_data = [all_landmarks[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_data = [all_landmarks[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(train_data, train_labels),
        batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        SignLandmarkDataset(val_data, val_labels),
        batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

    try:
        # Custom args for specific models
        model_kwargs = {
            'input_size': CONFIG['input_size'],
            'hidden_size': CONFIG['hidden_size'],
            'num_layers': CONFIG['num_layers'],
            'num_classes': len(label2idx),
            'dropout': CONFIG['dropout']
        }

        if model_name == 'transformer':
            model_kwargs.update({
                'nhead': CONFIG['nhead'],
                'dim_feedforward': CONFIG['dim_feedforward'],
                'max_len': CONFIG['max_len']
            })
        elif model_name == 'tcn':
            model_kwargs['kernel_size'] = CONFIG['tcn_kernel_size']

        model = model_class(**model_kwargs).to(device)

    except Exception as e:
        print(f"Model initialization error: {e}")
        traceback.print_exc()
        return

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()

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
            early_stopping_patience=CONFIG['early_stopping_patience']
        )

        log_experiment_result(model_name, best_val_acc, logs, CONFIG)

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
        ("transformer", SignTransformerClassifier),
        ("tcn", SignTCNClassifier)
    ]

    for model_name, model_class in model_variants:
        print(f"\n========== Training: {model_name} ==========")
        train(model_class, model_name)