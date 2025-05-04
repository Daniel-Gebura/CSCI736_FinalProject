################################################################
# models/gru_layernorm.py
#
# GRU classifier with Layer Normalization applied to final GRU output.
#
# Author: Daniel Gebura
################################################################

import torch
import torch.nn as nn

class SignGRUClassifier_LayerNorm(nn.Module):
    """
    GRU classifier with LayerNorm on the final hidden state.

    Args:
        input_size (int): Frame feature size.
        hidden_size (int): GRU hidden size per direction.
        num_layers (int): GRU layers.
        num_classes (int): Output classes.
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
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, x, lengths=None):
        _, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        norm = self.layer_norm(last_hidden)
        out = self.dropout(norm)
        return self.fc(out)