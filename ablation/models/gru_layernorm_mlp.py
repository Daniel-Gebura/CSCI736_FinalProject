################################################################
# models/gru_layernorm_mlp.py
#
# GRU classifier with both LayerNorm and a 2-layer MLP head.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

class SignGRUClassifier_LayerNorm_MLP(nn.Module):
    """
    GRU classifier using final hidden state + LayerNorm + deep MLP.

    Args:
        input_size (int): Frame input size.
        hidden_size (int): GRU hidden size.
        num_layers (int): Number of GRU layers.
        num_classes (int): Output classes.
        dropout (float): Dropout rate.
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
        _, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        norm = self.layer_norm(last_hidden)
        out = self.dropout(norm)
        return self.mlp(out)