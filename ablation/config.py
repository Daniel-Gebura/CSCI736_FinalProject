################################################################
# config.py
#
# Stores all configurable constants used for training,
# evaluation, and ablation experiments. Keeps the pipeline
# clean and makes it easy to run consistent experiments.
#
# v3: Increased dropout & weight decay, added early stopping patience
#     (based on Transformer overfitting)
#
# Author: Daniel Gebura
################################################################

CONFIG = {
    # ---------------- Randomization ----------------
    'seed': 42,

    # ---------------- Model Input -------------------
    'input_size': 132,

    # ------ Model Architecture (Example for Transformer/TCN) ------
    'hidden_size': 256,        # Transformer: d_model=256. TCN: num_channels=256.
    'num_layers': 4,           # Transformer: 4 layers. TCN: 4 layers.
    # --- MODIFIED --- Increased dropout
    'dropout': 0.5,

    # ---------------- Training Hyperparameters ----------------
    'batch_size': 32,
    'num_epochs': 250,         # Keep max epochs, early stopping will control duration
    'learning_rate': 0.0001,   # Kept lower LR from previous run
    # --- MODIFIED --- Increased weight decay
    'weight_decay': 0.0001,    # Increased L2 regularization (e.g., 1e-4)
    'validation_split': 0.2,
    # --- ADDED --- Early stopping patience
    'early_stopping_patience': 25, # Stop after 25 epochs with no val_acc improvement

    # ---------------- File Paths ----------------
    'zip_path': '../data/landmarks.zip',
    'landmarks_folder': 'landmarks/',
    'model_save_dir': './experiments/saved_models' # Make sure this path is correct
}