################################################################
# utils/metrics.py
#
# Contains training loop (`evaluate_model`) and seed control function.
# v3: Added early stopping logic.
#
# Author: Daniel Gebura
################################################################

import os
import json
import torch
import random
import numpy as np
from tqdm.auto import tqdm


def set_seed(seed):
    """
    Sets random seeds across libraries to ensure reproducibility.

    Args:
        seed (int): The seed to use for RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


def evaluate_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs, device,
                   save_dir, label2idx, idx2label, input_size_config, early_stopping_patience):
    """
    Full training and validation loop with best model checkpointing and early stopping.

    Args:
        model (nn.Module): Model instance to train.
        model_name (str): Unique model name string.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer instance.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to use.
        save_dir (str): Directory to store model files.
        label2idx (dict): Class label to index mapping.
        idx2label (dict): Index to class label mapping.
        input_size_config (int): Input size from the main config.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        float: Best validation accuracy achieved.
        dict: Logs of training losses/accuracies per epoch.
    """
    best_val_acc = 0.0
    epochs_no_improve = 0  # --- ADDED --- Counter for early stopping
    logs = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}

    model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    info_path = os.path.join(save_dir, f"{model_name}_info.json")

    for epoch in range(num_epochs):
        model.train()
        total_train, correct_train, train_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}/{num_epochs}")
        for x, y, lengths in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train
        logs['train_losses'].append(avg_train_loss)
        logs['train_accs'].append(train_acc)

        # --- Validation ---
        model.eval()
        total_val, correct_val, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(logits, 1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        logs['val_losses'].append(avg_val_loss)
        logs['val_accs'].append(val_acc)

        print(
            f"[Epoch {epoch + 1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")  # Added Val Loss print

        # Check for improvement and update early stopping counter
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0  # Reset counter
            torch.save(model.state_dict(), model_path)
            print(f"  -> Best model saved to {model_path} (Val Acc: {best_val_acc:.4f})")  # Added score to print

            # Get dropout rate safely
            dropout_rate = 0.0  # Default
            if hasattr(model, 'dropout') and hasattr(model.dropout, 'p'):
                dropout_rate = model.dropout.p
            elif hasattr(model, 'dropout_rate'):
                dropout_rate = model.dropout_rate

            # Get bidirectional status safely
            bidirectional_status = getattr(model, 'bidirectional', False)

            model_info_data = {
                'input_size': input_size_config,
                'hidden_size': getattr(model, 'hidden_size', None),
                'num_layers': getattr(model, 'num_layers', None),
                'bidirectional': bidirectional_status,
                'dropout': dropout_rate,
                'model_name': model_name,
                'architecture': model.__class__.__name__,
                'label_to_index': label2idx,
                'index_to_label': idx2label
            }
            model_info_data = {k: v for k, v in model_info_data.items() if v is not None}

            with open(info_path, 'w') as f:
                json.dump(model_info_data, f, indent=4)
        else:
            epochs_no_improve += 1  # Increment counter if no improvement

        # --- ADDED --- Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered after {epoch + 1} epochs ({early_stopping_patience} epochs without improvement).")
            print(f"Best Validation Accuracy: {best_val_acc:.4f}")
            break  # Exit the training loop

    return best_val_acc, logs
