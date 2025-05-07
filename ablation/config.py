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
    'seed': 42,  # Global seed for reproducibility

    # ---------------- Model Input -------------------
    'input_size': 132,  # Final input feature vector size (per frame)
                        # e.g., 63 + 63 + 3 + 3 for both hands + wrists

    # ---------------- Model Architecture ----------------
    'hidden_size': 128,        # GRU hidden state size (per direction)
    'num_layers': 2,           # Number of stacked GRU layers
    'dropout': 0.5,            # Dropout probability in GRU/MLP

    # ---------------- Training Hyperparameters ----------------
    'batch_size': 32,          # Batch size per gradient step
    'num_epochs': 200,          # Total number of training epochs
    'learning_rate': 0.001,    # Optimizer learning rate
    'weight_decay': 1e-5,      # L2 regularization (Adam weight decay)
    'validation_split': 0.2,   # Proportion of data used for validation

    # ---------------- File Paths ----------------
    'zip_path': '../data/landmarks.zip',            # Path to zipped dataset
    'landmarks_folder': 'landmarks/',              # Folder inside zip with .npy files
    'model_save_dir': './experiments/saved_models' # Directory for saving model weights and metadata
}