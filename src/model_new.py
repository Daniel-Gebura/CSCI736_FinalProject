import torch
import torch.nn as nn

# --- GRU Classifier + LayerNorm + Deeper MLP (Enhanced) ---
class SignGRUClassifier_LayerNorm_MLP(nn.Module):
    """
    GRU-based classifier with LayerNorm and a deeper MLP head.
    Now supports optional bidirectionality.

    Args:
        input_size (int): Size of input feature vector.
        hidden_size (int): Number of features in GRU hidden state (per direction).
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability for GRU (inter-layer) and MLP.
        bidirectional (bool): If True, use a bidirectional GRU. Default: False.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SignGRUClassifier_LayerNorm_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # Output features from GRU (accounts for bidirectionality)
        self.gru_output_features = hidden_size * 2 if bidirectional else hidden_size

        # GRU layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            # Apply dropout between layers only if num_layers > 1
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Layer Normalization - Applied to the combined GRU output features
        self.layer_norm = nn.LayerNorm(self.gru_output_features)

        # Dropout layer - Applied after LayerNorm before MLP
        self.dropout = nn.Dropout(dropout)

        # MLP head: gru_output_features -> hidden_size -> num_classes
        self.mlp = nn.Sequential(
            nn.Linear(self.gru_output_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )


        print("\n[Model Initialized: SignGRUClassifier_LayerNorm_MLP]")
        print(f"  • Input Size             : {input_size}")
        print(f"  • Hidden Size            : {hidden_size} (per GRU direction)")
        print(f"  • GRU Layers             : {num_layers}")
        print(f"  • Bidirectional GRU      : {bidirectional}")
        print(f"  • GRU Output Features    : {self.gru_output_features} (used for LayerNorm/MLP)")
        print(f"  • MLP Architecture       : {self.gru_output_features} → {hidden_size} → {hidden_size // 2} → {num_classes}")
        print(f"  • Dropout Rate           : {dropout}")
        print(f"  • Num Output Classes     : {num_classes}\n")



    def forward(self, x, lengths=None):
        """
        Forward pass through the model with LayerNorm and deeper MLP.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_size).
            lengths (torch.Tensor, optional): Sequence lengths (currently not used
                                              but kept for potential future use with packing).

        Returns:
            torch.Tensor: Output logits (batch_size, num_classes).
        """
        # x shape: (batch_size, seq_len, input_size)
        # GRU outputs:
        # gru_out shape: (batch_size, seq_len, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        gru_out, h_n = self.gru(x)

        if self.bidirectional:
            # Concatenate the final hidden states of the forward and backward GRU from the last layer
            # h_n is shaped (num_layers * 2, batch, hidden_size)
            # Get the last layer's forward state (index -2)
            last_hidden_fwd = h_n[-2, :, :]
            # Get the last layer's backward state (index -1)
            last_hidden_bwd = h_n[-1, :, :]
            # Concatenate along the feature dimension
            # Shape becomes (batch_size, hidden_size * 2)
            combined_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
        else:
            # Use the hidden state of the last layer (which is the last element of h_n for unidirectional)
            # Shape: (batch_size, hidden_size)
            combined_hidden = h_n[-1, :, :]

        # Apply Layer Normalization
        # Input shape: (batch_size, gru_output_features)
        normalized = self.layer_norm(combined_hidden)

        # Apply Dropout
        # Input shape: (batch_size, gru_output_features)
        dropped_out = self.dropout(normalized)

        # Pass through MLP
        # Input shape: (batch_size, gru_output_features)
        # Output shape: (batch_size, num_classes)
        out = self.mlp(dropped_out)

        return out

# ---------------------------------------------------------------------
# The other models updated similarly for completeness and consistency
# ---------------------------------------------------------------------

# --- GRU Classifier Model (Basic) ---
class SignGRUClassifier(nn.Module):
    """ Basic GRU Classifier with bidirectional option """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SignGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru_output_features = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(self.gru_output_features, num_classes)
        self.dropout = nn.Dropout(dropout) # Applied after getting hidden state

    def forward(self, x, lengths=None):
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden_fwd = h_n[-2, :, :]
            last_hidden_bwd = h_n[-1, :, :]
            combined_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
        else:
            combined_hidden = h_n[-1, :, :]
        out = self.dropout(combined_hidden) # Apply dropout before FC
        out = self.fc(out)
        return out

# --- GRU Classifier + Layer Normalization ---
class SignGRUClassifier_LayerNorm(nn.Module):
    """ GRU Classifier with LayerNorm and bidirectional option """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SignGRUClassifier_LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru_output_features = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(self.gru_output_features)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.gru_output_features, num_classes)

    def forward(self, x, lengths=None):
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden_fwd = h_n[-2, :, :]
            last_hidden_bwd = h_n[-1, :, :]
            combined_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
        else:
            combined_hidden = h_n[-1, :, :]
        normalized = self.layer_norm(combined_hidden)
        out = self.dropout(normalized)
        out = self.fc(out)
        return out

# --- GRU Classifier + Deeper MLP ---
class SignGRUClassifier_MLP(nn.Module):
    """ GRU Classifier with deeper MLP head and bidirectional option """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SignGRUClassifier_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru_output_features = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout) # Applied after getting hidden state
        # MLP head adjusted for bidirectional input and consistent intermediate size
        self.mlp = nn.Sequential(
            nn.Linear(self.gru_output_features, hidden_size), # Adjusted input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths=None):
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            last_hidden_fwd = h_n[-2, :, :]
            last_hidden_bwd = h_n[-1, :, :]
            combined_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
        else:
            combined_hidden = h_n[-1, :, :]
        out = self.dropout(combined_hidden) # Apply dropout before MLP
        out = self.mlp(out)
        return out