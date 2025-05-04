################################################################
# models/tcn_classifier.py
#
# Temporal Convolutional Network (TCN) based classifier for
# sign language landmarks.
#
# Reference implementation structure:
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
#
# v2: Added input_size attribute for compatibility with saving metadata.
#
# Author: Gemini
################################################################

import torch
import torch.nn as nn
# from torch.nn.utils import weight_norm
# NOTE: Using weight_norm requires PyTorch >= 1.11 or installing via parametrizations package
# If you have issues, comment out the import above and the weight_norm calls below
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Removes the extra padding added by causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Input shape: (Batch, Channels, Time)
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single block of temporal convolution"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Use weight_norm for potentially better/faster convergence
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # Remove padding to make it causal
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual connection if dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # Initialize convolutional layers
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
             nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """The core TCN network"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs (int): Number of input features per time step.
            num_channels (list): List containing the number of channels for each TCN layer.
                                 Length determines the number of layers.
            kernel_size (int): Convolution kernel size.
            dropout (float): Dropout rate.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Calculate padding based on dilation and kernel size for causal convolution
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (Batch, Features, Time) - Note the channel dimension difference!
        return self.network(x)


class SignTCNClassifier(nn.Module):
    """
    TCN-based sequence classifier.

    Args:
        input_size (int): Feature vector size per frame.
        hidden_size (int): Size of the hidden channels in TCN blocks. Can be a single
                           int (repeated for all layers) or a list of ints.
                           For simplicity, using hidden_size from config as base.
        num_layers (int): Number of TCN blocks/layers.
        num_classes (int): Output class count.
        kernel_size (int): Kernel size for convolutions. Default: 3.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size=3, dropout=0.5):
        super().__init__()

        # --- ADDED --- Store input_size for compatibility with saving metadata
        self.input_size = input_size
        self.hidden_size = hidden_size # Store for metadata saving
        self.num_layers = num_layers # Store for metadata saving
        self.dropout_rate = dropout # Store dropout rate
        # -------------

        # Define number of channels for each layer
        # Example: [hidden_size] * num_layers means all layers have hidden_size channels
        # You could also define a more complex structure, e.g., increasing channels
        tcn_channels = [hidden_size] * num_layers
        self.output_dim = tcn_channels[-1] # Output dimension from TCN

        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size=kernel_size, dropout=dropout)

        # Classifier head
        self.dropout_final = nn.Dropout(dropout) # Use stored rate
        self.fc = nn.Linear(self.output_dim, num_classes)

        # --- ADDED --- Define attributes needed for saving metadata
        self.bidirectional = False # TCN is causal, not bidirectional
        # Need a dropout layer instance to access .p if metrics.py uses it
        # If metrics.py only uses self.dropout_rate, this isn't strictly needed
        self.dropout = self.dropout_final

    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_size)

        # TCN expects input shape: (batch_size, num_features, seq_len)
        x = x.transpose(1, 2) # -> (batch_size, input_size, seq_len)

        # Pass through TCN
        tcn_output = self.tcn(x) # -> (batch_size, output_dim, seq_len)

        # Use the output of the last time step for classification
        # Alternatively, use global average pooling: tcn_output.mean(dim=2)
        last_time_step_output = tcn_output[:, :, -1] # -> (batch_size, output_dim)

        dropped_output = self.dropout_final(last_time_step_output)

        # Final classification
        logits = self.fc(dropped_output) # -> (batch_size, num_classes)
        return logits