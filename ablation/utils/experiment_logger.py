################################################################
# utils/experiment_logger.py
#
# Logs model performance and hyperparameters into a .csv.
# Helps track results from all ablation runs.
#
# Author: Daniel Gebura
################################################################

import os
import csv
from datetime import datetime


def log_experiment_result(model_name, best_val_acc, logs, config, log_path="experiments/results.csv"):
    """
    Appends experiment metrics to results CSV file.

    Args:
        model_name (str): Identifier for the model variant.
        best_val_acc (float): Best validation accuracy reached.
        logs (dict): Contains train/val losses and accs.
        config (dict): All relevant config and hyperparams.
        log_path (str): Path to CSV file to log results.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Compose log row
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'val_acc': f"{best_val_acc:.4f}",
        'train_acc_last': f"{logs['train_accs'][-1]:.4f}",
        'val_acc_last': f"{logs['val_accs'][-1]:.4f}",
        'input_size': config['input_size'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'epochs': config['num_epochs'],
        'batch_size': config['batch_size'],
        'lr': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'seed': config['seed']
    }

    # If file does not exist, write header
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Log] Results for {model_name} appended to {log_path}")
