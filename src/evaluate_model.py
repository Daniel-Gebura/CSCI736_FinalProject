################################################################
# evaluate_model.py
#
# Loads the best saved model and evaluates it on validation data.
# Generates a confusion matrix heatmap and saves a report
# of the most confused class pairs.
#
# Author: Daniel Gebura
################################################################

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataloader import load_landmark_data_from_zip, SignLandmarkDataset, collate_fn
from model import SignGRUClassifierAttention

# ---------------- Configuration ----------------

# Paths
ZIP_FILE_PATH = '../data/landmarks.zip'
LANDMARKS_FOLDER_IN_ZIP = 'landmarks/'
MODEL_SAVE_PATH = '../saved_models/sign_gru_classifier_best.pth'
MODEL_INFO_PATH = '../saved_models/sign_gru_classifier_info.json'

# Output
HEATMAP_PATH = 'confusion_matrix_heatmap.png'
REPORT_PATH = 'most_confused_report.txt'
BATCH_SIZE = 64

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------

def load_model_and_info(model_path, info_path):
    """
    Load the trained model and configuration.

    Args:
        model_path (str): Path to model weights.
        info_path (str): Path to model metadata.

    Returns:
        model (nn.Module), label2idx (dict), idx2label (dict)
    """
    with open(info_path, 'r') as f:
        info = json.load(f)

    model = SignGRUClassifierAttention(
        input_size=info['input_size'],
        hidden_size=info['hidden_size'],
        num_layers=info['num_layers'],
        num_classes=len(info['label_to_index']),
        dropout=info['dropout'],
        bidirectional=info['bidirectional']
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, info['label_to_index'], info['index_to_label']

def evaluate(model, dataloader):
    """
    Run model predictions on a dataloader.

    Returns:
        Tuple of lists: true labels, predicted labels
    """
    y_true, y_pred = [], []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

def plot_confusion_heatmap(y_true, y_pred, idx2label, output_path):
    """
    Plot and save a heatmap of the confusion matrix.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        idx2label (dict): Index to label mapping.
        output_path (str): Path to save the heatmap image.
    """
    label_ids = list(map(int, idx2label.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize='true')

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, cmap='viridis', cbar=True)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def report_most_confused(y_true, y_pred, idx2label, output_path, top_n=10):
    """
    Identify and save a report of the most confused class pairs.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        idx2label (dict): Index to label mapping.
        output_path (str): Path to save the report.
        top_n (int): Number of most confused pairs to report.
    """
    label_ids = list(map(int, idx2label.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    np.fill_diagonal(cm, 0)  # Zero out diagonal to focus on confusion

    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confused_pairs.append((cm[i, j], idx2label[str(i)], idx2label[str(j)]))

    confused_pairs.sort(reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Top Most Confused Class Pairs (True → Predicted):\n\n")
        for count, true_label, pred_label in confused_pairs[:top_n]:
            f.write(f"{true_label:30} → {pred_label:30} | Count: {count}\n")

    print(f"Most confused class report saved to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Evaluating trained model...\n")

    # Load dataset
    all_landmarks, all_labels, label2idx, idx2label, fnames = load_landmark_data_from_zip(ZIP_FILE_PATH, LANDMARKS_FOLDER_IN_ZIP)

    val_dataset = SignLandmarkDataset(all_landmarks, all_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load model and mappings
    model, label2idx, idx2label = load_model_and_info(MODEL_SAVE_PATH, MODEL_INFO_PATH)

    # Evaluate
    y_true, y_pred = evaluate(model, val_loader)

    # Visualize confusion matrix
    plot_confusion_heatmap(y_true, y_pred, idx2label, HEATMAP_PATH)

    # Generate most confused report
    report_most_confused(y_true, y_pred, idx2label, REPORT_PATH)