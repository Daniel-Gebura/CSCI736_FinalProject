import torch
import torch.nn as nn

class SignGRUClassifier_LayerNorm_MLP_Bi_Attention(nn.Module):
    """
    A standalone GRU-based sequence classifier for gesture recognition.
    This model uses:
      - A configurable GRU (optionally bidirectional)
      - A self-attention mechanism over GRU outputs
      - LayerNorm and Dropout for regularization
      - A multi-layer MLP classifier head

    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, num_classes)
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        """
        Initializes the model layers.

        Args:
            input_size (int): Size of input feature vector for each frame.
            hidden_size (int): GRU hidden size per direction.
            num_layers (int): Number of stacked GRU layers.
            num_classes (int): Number of output gesture classes.
            dropout (float): Dropout probability used in GRU and MLP layers.
            bidirectional (bool): Whether to use a bidirectional GRU. Default is False.
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
        self.layer_norm = nn.LayerNorm(self.gru_output_features)  # Normalize attention-weighted sum
        self.dropout = nn.Dropout(dropout)                         # Applied after LayerNorm

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
        normalized = self.layer_norm(context_vector)  # (batch, hidden_dim)
        dropped = self.dropout(normalized)

        # --- Step 4: MLP Classification ---
        logits = self.mlp(dropped)  # Final shape: (batch, num_classes)

        return logits


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