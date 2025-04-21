################################################################
# model.py
#
# Defines model architectures for training.
# Flexible to allow importing different models from here.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

# --- GRU Classifier Model ---
class SignGRUClassifier(nn.Module):
    """
    GRU-based classifier for sequence classification tasks.

    Args:
        input_size (int): Size of input feature vector (e.g., 126 for hand landmarks).
        hidden_size (int): Number of features in GRU hidden state.
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of target classes for classification.
        dropout (float): Dropout probability after GRU layers.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer(s)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Fully-connected classification head
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        """
        Forward pass for the GRU classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            lengths (torch.Tensor, optional): Sequence lengths for packed input (not used here).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # GRU outputs
        gru_out, h_n = self.gru(x)

        # Take the hidden state of the last GRU layer
        last_hidden_state = h_n[-1]  # Shape: (batch_size, hidden_size)

        # Apply dropout and final linear layer
        out = self.dropout(last_hidden_state)
        out = self.fc(out)  # Shape: (batch_size, num_classes)

        return out
