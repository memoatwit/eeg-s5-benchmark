"""
CNN Model for EEG Decoding
===========================
Convolutional Neural Network classifier for naturalistic EEG decoding.
Strong baseline — matches S5 accuracy at 64s but uses ~20x more parameters.
"""

import torch.nn as nn


class CNNClassifier(nn.Module):
    """CNN-based EEG Classifier"""

    def __init__(self, input_dim=64, hidden_dim=128, n_classes=3, dropout=0.1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=25, padding=12),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=25, padding=12),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.convs(x)
        x = x.squeeze(-1)
        return self.classifier(x)
