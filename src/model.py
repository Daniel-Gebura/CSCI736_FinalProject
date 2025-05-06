################################################################
# model.py
#
# Defines model architecture for gesture recognition using a GRU-based
# classifier.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

class SignGRUClassifierAttention(nn.Module):
    """
    A standalone GRU-based sequence classifier for gesture recognition.
    This model uses:
      - A configurable GRU (optionally bidirectional)
      - A self-attention mechanism over GRU outputs
      - Dropout for regularization
      - A multi-layer MLP classifier head

    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, num_classes)
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        """
        Initializes the model layers.

        Args:
            input_size (int): Size of input feature vector for each frame.
            hidden_size (int): GRU hidden size per direction.
            num_layers (int): Number of stacked GRU layers.
            num_classes (int): Number of output gesture classes.
            dropout (float): Dropout probability used in GRU and MLP layers.
            bidirectional (bool): Whether to use a bidirectional GRU. Default is True.
        """
        super().__init__()

        # Save model config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Total GRU output features (doubles if bidirectional)
        self.gru_output_features = hidden_size * 2 if bidirectional else hidden_size

        # --- GRU Encoder ---
        # Processes the input sequence frame-by-frame
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # --- Attention Mechanism ---
        # Maps GRU output at each time step to a scalar attention score
        self.attention_fc = nn.Linear(self.gru_output_features, 1)

        # --- Regularization ---
        self.dropout = nn.Dropout(dropout)

        # --- MLP Classifier Head ---
        self.mlp = nn.Sequential(
            nn.Linear(self.gru_output_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # --- Debug Info ---
        print("\n[Initialized: SignGRUClassifierAttention]")
        print(f"  • Input Size             : {input_size}")
        print(f"  • Hidden Size            : {hidden_size} (per GRU direction)")
        print(f"  • GRU Layers             : {num_layers}")
        print(f"  • Bidirectional GRU      : {bidirectional}")
        print(f"  • GRU Output Features    : {self.gru_output_features} (used for attention)")
        print(f"  • MLP Architecture       : {self.gru_output_features} → {hidden_size} → {hidden_size // 2} → {num_classes}")
        print(f"  • Dropout Rate           : {dropout}")
        print(f"  • Num Output Classes     : {num_classes}\n")

    def forward(self, x, lengths=None):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)
            lengths (torch.Tensor, optional): Original lengths of each sequence (unused)

        Returns:
            torch.Tensor: Logits for each class (batch, num_classes)
        """
        # --- Step 1: GRU Encoding ---
        # GRU processes each frame; returns all hidden states (one per frame)
        # Output shape: (batch, seq_len, hidden_size * num_directions)
        gru_out, _ = self.gru(x)

        # --- Step 2: Attention Mechanism ---
        # Compute scalar attention scores for each time step
        # Shape: (batch, seq_len, 1) → squeeze → (batch, seq_len)
        attn_scores = self.attention_fc(gru_out).squeeze(-1)

        # Normalize scores using softmax to get attention weights
        # Shape: (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Apply attention weights to each time step and sum
        # Broadcast weights over feature dim and sum across time axis
        # Shape: (batch, seq_len, hidden_dim) * (batch, seq_len, 1) → sum over seq_len
        context_vector = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)

        # --- Step 3: Normalize and Dropout ---
        dropped = self.dropout(context_vector)

        # --- Step 4: MLP Classification ---
        logits = self.mlp(dropped)  # Final shape: (batch, num_classes)

        return logits