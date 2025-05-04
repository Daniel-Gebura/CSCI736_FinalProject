################################################################
# utils/metrics.py
#
# Contains training loop (`evaluate_model`) and seed control function.
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
                   save_dir, label2idx, idx2label):
    """
    Full training and validation loop with best model checkpointing.

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

    Returns:
        float: Best validation accuracy achieved.
        dict: Logs of training losses/accuracies per epoch.
    """
    best_val_acc = 0.0
    logs = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}

    model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    info_path = os.path.join(save_dir, f"{model_name}_info.json")

    for epoch in range(num_epochs):
        model.train()
        total_train, correct_train, train_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}")
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

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save model if performance improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> Best model saved to {model_path}")
            # Save metadata
            with open(info_path, 'w') as f:
                json.dump({
                    'input_size': model.gru.input_size,
                    'hidden_size': model.hidden_size,
                    'num_layers': model.num_layers,
                    'bidirectional': model.bidirectional,
                    'dropout': model.dropout.p,
                    'label_to_index': label2idx,
                    'index_to_label': idx2label
                }, f, indent=4)

    return best_val_acc, logs