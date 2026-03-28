"""
LSTM Model for EEG Decoding
============================
Bidirectional LSTM classifier for naturalistic EEG decoding.
"""

import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM-based EEG Classifier"""

    def __init__(self, input_dim=64, hidden_dim=128, n_layers=2, n_classes=3, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        output, (hidden, cell) = self.lstm(x)
        x = output[:, -1, :]  # Use last timestep output
        return self.classifier(x)
