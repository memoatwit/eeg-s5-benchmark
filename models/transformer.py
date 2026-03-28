"""
Transformer Model for EEG Decoding
=====================================
Standard Transformer classifier for naturalistic EEG decoding.
See eeg_transformer.py for EEGXF — the stabilized EEG-specific variant
introduced in this paper.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """Transformer-based EEG Classifier"""

    def __init__(self, input_dim=64, d_model=256, nhead=8, num_layers=4,
                 n_classes=3, dropout=0.1, max_seq_len=4000):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        batch_size, seq_len, input_dim = x.shape

        if seq_len == 0:
            logger.warning("Transformer received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)

        x = self.input_projection(x)

        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :]
        else:
            repeat_factor = (seq_len // self.positional_encoding.size(1)) + 1
            pos_enc = self.positional_encoding.repeat(1, repeat_factor, 1)[:, :seq_len, :]

        x = x + pos_enc
        x = self.transformer(x)
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        return self.classifier(x)
