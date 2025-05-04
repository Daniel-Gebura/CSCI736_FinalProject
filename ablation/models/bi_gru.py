################################################################
# models/bi_gru.py
#
# Bidirectional GRU classifier using the final hidden state for classification.
# No LayerNorm, attention, or deep MLP.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

class SignBiGRUClassifier(nn.Module):
    """
    GRU-based sequence classifier using only final hidden state.

    Args:
        input_size (int): Feature vector size per frame.
        hidden_size (int): GRU hidden state size.
        num_layers (int): Number of GRU layers.
        num_classes (int): Output class count.
        dropout (float): Dropout rate.
        bidirectional (bool): If True, use bidirectional GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, x, lengths=None):
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)