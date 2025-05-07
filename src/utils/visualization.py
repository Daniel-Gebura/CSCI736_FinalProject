################################################################
# utils/visualization.py
#
# Contains plotting utilities to visualize training/validation metrics.
#
# Author: Daniel Gebura
################################################################

import matplotlib.pyplot as plt
import os


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    """
    Saves plots of loss and accuracy over epochs.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accs (list): Training accuracy per epoch.
        val_accs (list): Validation accuracy per epoch.
        model_name (str): Used to name output file.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = f"training_curves_{model_name}.png"
    plt.savefig(save_path)
    print(f"[Plot] Training curves saved to {save_path}")