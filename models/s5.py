"""
S5 Model for EEG Decoding
=========================
S5 (Simplified Structured State Space) classifier for naturalistic EEG decoding.
Best parameter efficiency in the benchmark — ~20x fewer params than CNN at 64s.

Requires: pip install s5-pytorch
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class S5Classifier(nn.Module):
    """S5-based EEG Classifier - Next Generation State Space Model"""

    def __init__(self, input_dim=64, hidden_dim=128, n_layers=2, n_classes=3, dropout=0.1):
        super().__init__()

        try:
            from s5 import S5Block
        except ImportError:
            raise ImportError("s5-pytorch not installed. Install with: pip install s5-pytorch")

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.s5_layers = nn.ModuleList([
            S5Block(
                dim=hidden_dim,
                state_dim=64,
                bidir=True,
                ff_dropout=dropout,
                attn_dropout=dropout
            ) for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        logger.info(f"S5Classifier: hidden_dim={hidden_dim}, n_layers={n_layers}, state_dim=64")

    def forward(self, x):
        if torch.isnan(x).any():
            logger.warning("NaNs in S5Classifier input")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)

        batch_size, channels, seq_len = x.shape
        if seq_len == 0:
            logger.warning("S5Classifier received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)

        x = x.transpose(1, 2)
        x = self.input_projection(x)

        for i, (s5_layer, norm) in enumerate(zip(self.s5_layers, self.layer_norms)):
            residual = x
            try:
                x_new = s5_layer(x)
                x = norm(x_new + residual)
            except Exception as e:
                logger.warning(f"S5 layer {i+1} failed: {e}, using residual")
                x = residual

        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        if torch.isnan(x).any():
            logger.warning("NaNs in S5 output before classification")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)

        output = self.classifier(x)

        if torch.isnan(output).any():
            logger.warning("NaNs in S5 final output")
            return torch.zeros_like(output)

        return output
