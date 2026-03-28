"""
S4 Model for EEG Decoding
=========================
Optimized S4 (Structured State Space) classifier for naturalistic EEG decoding.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OptimizedS4Layer(nn.Module):
    """Optimized S4 Layer with reduced state size and improved efficiency"""

    def __init__(self, d_model, d_state=16, dropout=0.1, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(d_state) * 0.0001)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.001)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.001)
        self.D = nn.Parameter(torch.randn(d_model) * 0.0001)

        self.register_buffer('dt', torch.tensor(0.0001))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        print(f"OptimizedS4Layer initialized: d_model={d_model}, d_state={d_state} (reduced for efficiency)")

    def get_kernel(self, length=1000):
        """Extract the convolution kernel for visualization"""
        if length <= 0:
            logger.warning(f"Invalid kernel length {length} - using default 1000")
            length = 1000

        with torch.no_grad():
            A_discrete = torch.exp(-self.dt * torch.abs(self.A))
            B_discrete = self.dt * self.B

            kernels = []
            h = torch.zeros(self.d_state, device=self.A.device)

            for t in range(length):
                u = torch.ones(self.d_model, device=self.A.device) if t == 0 \
                    else torch.zeros(self.d_model, device=self.A.device)
                h = h * A_discrete + torch.sum(B_discrete * u.unsqueeze(0), dim=1)
                y = torch.sum(self.C * h.unsqueeze(0), dim=1) + self.D * u
                kernels.append(y)

            return torch.stack(kernels, dim=0)

    def get_frequency_response(self, freqs, sampling_rate=250):
        """Compute frequency response for visualization"""
        with torch.no_grad():
            kernel = self.get_kernel(length=1000)
            freq_responses = []
            for dim in range(self.d_model):
                kernel_1d = kernel[:, dim].cpu().numpy()
                kernel_padded = np.pad(kernel_1d, (0, 2048 - len(kernel_1d)), 'constant')
                fft = np.fft.fft(kernel_padded)
                freq_responses.append(np.abs(fft[:len(freqs)]))
            return np.array(freq_responses)

    def forward(self, x):
        """S4 forward pass"""
        batch_size, seq_len, d_model = x.shape

        if seq_len == 0:
            logger.warning("S4 received zero sequence length - returning zeros")
            return torch.zeros(batch_size, 0, d_model, device=x.device)

        if torch.isnan(x).any():
            logger.warning("NaNs found in S4 input - replacing with zeros")
            x = torch.zeros_like(x)

        A_discrete = torch.exp(-self.dt * torch.abs(self.A))
        B_discrete = self.dt * self.B

        if torch.isnan(A_discrete).any() or torch.isnan(B_discrete).any():
            logger.warning("NaNs in S4 discrete system - using fallback")
            return torch.zeros_like(x)

        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            u_t = x[:, t, :]
            h_next = h * A_discrete.unsqueeze(0)
            B_u = torch.matmul(B_discrete, u_t.T).T
            h = torch.clamp(h_next + B_u, min=-10, max=10)
            C_h = torch.matmul(self.C, h.T).T
            D_u = u_t * self.D.unsqueeze(0)
            y_t = torch.clamp(C_h + D_u, min=-10, max=10)

            if torch.isnan(y_t).any():
                logger.warning(f"NaNs in S4 output at step {t}")
                y_t = torch.zeros_like(y_t)

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)

        if torch.isnan(output).any():
            logger.warning("NaNs in final S4 output - replacing")
            output = torch.zeros_like(output)

        return self.dropout(self.norm(output))


class OptimizedS4Classifier(nn.Module):
    """Optimized S4-based EEG Classifier with efficiency improvements"""

    def __init__(self, input_dim=64, d_model=128, d_state=16, n_layers=1,
                 n_classes=3, dropout=0.1, max_seq_len=4000, downsample_factor=2):
        super().__init__()

        print(f"OptimizedS4Classifier init: input_dim={input_dim}, d_model={d_model}, n_classes={n_classes}")
        print(f"  Efficiency optimizations: d_state={d_state}, n_layers={n_layers}, downsample={downsample_factor}x")

        self.downsample_factor = downsample_factor

        if downsample_factor > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=downsample_factor, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Identity()

        self.input_projection = nn.Linear(input_dim, d_model)

        max_downsampled_len = max_seq_len // downsample_factor
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_downsampled_len, d_model) * 0.0001
        )

        self.s4_layers = nn.ModuleList([
            OptimizedS4Layer(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        logger.info(f"OptimizedS4Classifier: d_model={d_model}, d_state={d_state}, n_layers={n_layers}")

    def forward(self, x):
        if torch.isnan(x).any():
            logger.warning("NaNs in OptimizedS4Classifier input")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)

        batch_size, channels, seq_len = x.shape
        if seq_len == 0:
            logger.warning("S4Classifier received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)

        x_downsampled = self.downsample(x)

        if x_downsampled.size(-1) == 0:
            logger.warning("S4Classifier downsampling resulted in zero length - using smaller downsample")
            fallback_downsample = max(1, seq_len // 10)
            x_downsampled = F.avg_pool1d(x, kernel_size=fallback_downsample, stride=fallback_downsample)

        x = x_downsampled.transpose(1, 2)
        x = self.input_projection(x)

        if torch.isnan(x).any():
            logger.warning("NaNs after input projection")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)

        seq_len = x.size(1)
        if seq_len == 0:
            logger.warning("S4Classifier sequence length is zero after downsampling")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)

        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :]
        else:
            repeat_factor = (seq_len // self.positional_encoding.size(1)) + 1
            pos_enc = self.positional_encoding.repeat(1, repeat_factor, 1)[:, :seq_len, :]

        x = x + pos_enc

        for i, layer in enumerate(self.s4_layers):
            residual = x
            x_new = layer(x)
            if torch.isnan(x_new).any():
                logger.warning(f"NaNs from OptimizedS4 layer {i+1}, using residual only")
                x = residual
            else:
                x = self.layer_norms[i](F.relu(residual + x_new))

        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        if torch.isnan(x).any():
            logger.warning("NaNs after global pooling")
            x = torch.zeros_like(x)

        output = self.classifier(x)

        if torch.isnan(output).any():
            logger.warning("NaNs in final OptimizedS4 output")
            output = torch.zeros_like(output)

        return output


# Alias for backward compatibility
S4Classifier = OptimizedS4Classifier
