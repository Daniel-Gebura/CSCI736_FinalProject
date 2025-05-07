################################################################
# models/transformer_classifier.py
#
# Transformer Encoder based classifier for sign language landmarks.
# Uses positional encoding and classifies based on the output
# of the first time step after passing through the encoder.
#
# Author: Sayantan Saha
################################################################

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Input x is batch_first, need to transpose for PE addition
        x = x.transpose(0, 1) # shape [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.transpose(0, 1) # shape [batch_size, seq_len, embedding_dim]
        return x

class SignTransformerClassifier(nn.Module):
    """
    Transformer Encoder based sequence classifier.

    Args:
        input_size (int): Feature vector size per frame.
        hidden_size (int): Dimension of the transformer model (d_model).
                            Should be divisible by nhead.
        num_layers (int): Number of Transformer Encoder layers.
        num_classes (int): Output class count.
        nhead (int): Number of heads in multiheadattention models.
                     Default: 4 (ensure hidden_size is divisible by 4)
        dim_feedforward (int): Dimension of the feedforward network model.
                               Default: 512
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length for positional encoding.
                       Default: 500 (adjust if sequences are longer)
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 nhead=4, dim_feedforward=512, dropout=0.5, max_len=500):
        super().__init__()

        if hidden_size % nhead != 0:
            original_hidden_size = hidden_size
            hidden_size = (hidden_size // nhead) * nhead
            if hidden_size == 0: hidden_size = nhead
            print(f"Warning: hidden_size {original_hidden_size} not divisible by nhead {nhead}. Adjusted hidden_size to {hidden_size}.")
            if hidden_size == 0:
                 raise ValueError("Hidden size resulted in 0 after adjustment. Please check hidden_size and nhead.")


        self.d_model = hidden_size
        # --- ADDED --- Store input_size for compatibility with saving metadata
        self.input_size = input_size
        self.hidden_size = hidden_size # Store for metadata saving
        self.num_layers = num_layers # Store for metadata saving
        self.dropout_rate = dropout # Store dropout rate
        # -------------

        # Linear layer to project input_size to d_model (hidden_size)
        self.input_proj = nn.Linear(input_size, self.d_model)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(self.d_model) # Optional: Apply LayerNorm before classifier
        self.dropout = nn.Dropout(dropout) # Use stored rate

        # Classifier head (similar to your MLP)
        # Make sure hidden_size is accessible for the MLP part
        mlp_hidden = self.hidden_size // 2 if self.hidden_size > 1 else 1 # Ensure positive dim
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout), # Use stored rate
            nn.Linear(mlp_hidden, num_classes)
        )
        # --- ADDED --- Define attributes needed for saving metadata
        # Transformer doesn't have 'bidirectional' in the same sense as GRU
        self.bidirectional = False # Set to False or None for compatibility

        # --- Debug Info ---
        print(f"""
        [Initialized: SignTransformerClassifier]
        • Input Size               : {input_size}
        • Hidden Size (d_model)    : {hidden_size}
        • Num Encoder Layers       : {num_layers}
        • Num Attention Heads      : {nhead}
        • Feedforward Dim          : {dim_feedforward}
        • Dropout Rate             : {dropout}
        • Max Sequence Length      : {max_len}
        • MLP Classifier Head      : {hidden_size} → {hidden_size // 2} → {num_classes}
        • Num Output Classes       : {num_classes}
        """)


    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_size) -> Should be on target device (e.g., cuda)

        # Project input features to model dimension
        x = self.input_proj(x) * math.sqrt(self.d_model) # Scale input embedding

        # Add positional encoding
        x = self.pos_encoder(x) # shape: (batch_size, seq_len, d_model)

        # Create mask for padding (optional but recommended)
        # TransformerEncoderLayer expects src_key_padding_mask: (batch_size, seq_len)
        # True indicates a padded position.
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            batch_size = x.size(0)
            # --- FIX: Move lengths to the same device as x ---
            lengths = lengths.to(x.device)
            # --------------------------------------------------
            # Create mask: True for indices >= length
            src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]

        # Pass through transformer encoder
        encoded_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # encoded_output shape: (batch_size, seq_len, d_model)

        # Use the output of the first token/time step for classification
        # Alternatively, use mean pooling over time: encoded_output.mean(dim=1)
        class_token_output = encoded_output[:, 0, :] # shape: (batch_size, d_model)

        # Optional LayerNorm
        norm_output = self.layer_norm(class_token_output)
        dropped_output = self.dropout(norm_output)

        # Final classification
        logits = self.fc(dropped_output) # shape: (batch_size, num_classes)
        return logits