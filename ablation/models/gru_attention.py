################################################################
# models/gru_attention.py
#
# GRU classifier with attention mechanism over time steps.
# Attention applied before MLP head.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

class SignGRUClassifierAttention(nn.Module):
    """
    GRU classifier with self-attention across time and MLP head.

    Args:
        input_size (int): Input feature size per frame.
        hidden_size (int): GRU hidden state size.
        num_layers (int): Number of GRU layers.
        num_classes (int): Number of gesture classes.
        dropout (float): Dropout probability.
        bidirectional (bool): Use bidirectional GRU if True.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        self.attn_fc = nn.Linear(self.output_dim, 1)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)  # (batch, seq_len, output_dim)
        attn_scores = self.attn_fc(gru_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, output_dim)
        norm = self.layer_norm(context)
        dropped = self.dropout(norm)
        return self.mlp(dropped)