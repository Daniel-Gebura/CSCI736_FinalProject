################################################################
# config.py
#
# Stores all configurable constants used for training,
# evaluation, and ablation experiments. Keeps the pipeline
# clean and makes it easy to run consistent experiments.
#
# Author: Daniel Gebura
################################################################

CONFIG = {
    # ---------------- Randomization ----------------
    'seed': 42,

    # ---------------- Model Input -------------------
    'input_size': 132,  # Set based on dataset (63 + 63 + 3 + 3)

    # ---------------- Architecture ------------------
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.5,

    # Transformer specific
    'nhead': 4,
    'dim_feedforward': 512,
    'max_len': 500,

    # TCN specific
    'tcn_kernel_size': 3,

    # ---------------- Training Hyperparameters ----------------
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'early_stopping_patience': 25,

    # ---------------- File Paths ----------------
    'zip_path': '../data/landmarks.zip',
    'landmarks_folder': 'landmarks/',
    'model_save_dir': '../saved_models'
}