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

# ---------------------------------------------------------------------

# --- GRU Classifier + Layer Normalization ---
class SignGRUClassifier_LayerNorm(nn.Module):
    """
    GRU-based classifier with LayerNorm after GRU output.

    Args:
        input_size (int): Size of input feature vector.
        hidden_size (int): Number of features in GRU hidden state.
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignGRUClassifier_LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass through the model with LayerNorm.

        Args:
            x (torch.Tensor): Input tensor.
            lengths (torch.Tensor, optional): Sequence lengths (not used).

        Returns:
            torch.Tensor: Output logits.
        """
        gru_out, h_n = self.gru(x)
        last_hidden_state = h_n[-1]
        normalized = self.layer_norm(last_hidden_state)
        out = self.dropout(normalized)
        out = self.fc(out)
        return out

# ---------------------------------------------------------------------

# --- GRU Classifier + Deeper MLP ---
class SignGRUClassifier_MLP(nn.Module):
    """
    GRU-based classifier with a deeper MLP head.

    Args:
        input_size (int): Size of input feature vector.
        hidden_size (int): Number of features in GRU hidden state.
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignGRUClassifier_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # MLP head: hidden_size -> hidden_size//2 -> output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        Forward pass through the model with deeper MLP.

        Args:
            x (torch.Tensor): Input tensor.
            lengths (torch.Tensor, optional): Sequence lengths (not used).

        Returns:
            torch.Tensor: Output logits.
        """
        gru_out, h_n = self.gru(x)
        last_hidden_state = h_n[-1]
        out = self.dropout(last_hidden_state)
        out = self.mlp(out)
        return out

# ---------------------------------------------------------------------

# --- GRU Classifier + LayerNorm + Deeper MLP ---
class SignGRUClassifier_LayerNorm_MLP(nn.Module):
    """
    GRU-based classifier with LayerNorm and a deeper MLP head.

    Args:
        input_size (int): Size of input feature vector.
        hidden_size (int): Number of features in GRU hidden state.
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignGRUClassifier_LayerNorm_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # MLP head: hidden_size -> hidden_size//2 -> output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        Forward pass through the model with LayerNorm and deeper MLP.

        Args:
            x (torch.Tensor): Input tensor.
            lengths (torch.Tensor, optional): Sequence lengths (not used).

        Returns:
            torch.Tensor: Output logits.
        """
        gru_out, h_n = self.gru(x)
        last_hidden_state = h_n[-1]
        normalized = self.layer_norm(last_hidden_state)
        out = self.dropout(normalized)
        out = self.mlp(out)
        return out
