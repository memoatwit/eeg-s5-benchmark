#!/usr/bin/env python3
"""
ICASSP EEG Benchmarking Script: Overnight Training
==================================================
Comprehensive evaluation of S4, Transformer, CNN, LSTM models on HBN-EEG movie data.

Author: Embedded AI Research Team
Date: July 6, 2025
Purpose: ICASSP 2025 Submission - Long-range Temporal Modeling in Naturalistic EEG
"""

import sys
import os
import time
import logging
import random
from pathlib import Path
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import mne
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import EMA (optional)
try:
    from torch_ema import ExponentialMovingAverage
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    print("Warning: torch_ema not available. Install with: pip install torch_ema")

# Suppress MNE info messages for cleaner output
mne.set_log_level('WARNING')

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed} for reproducibility")

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class OptimizedS4Layer(nn.Module):
    """Optimized S4 Layer with reduced state size and improved efficiency"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Optimized: Reduced state size for faster computation
        # Conservative initialization for stability
        self.A = nn.Parameter(torch.randn(d_state) * 0.0001)  # Even smaller init
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.001)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.001)
        self.D = nn.Parameter(torch.randn(d_model) * 0.0001)
        
        # More conservative time step for stability
        self.register_buffer('dt', torch.tensor(0.0001))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        print(f"OptimizedS4Layer initialized: d_model={d_model}, d_state={d_state} (reduced for efficiency)")
    
    def get_kernel(self, length=1000):
        """Extract the convolution kernel for visualization"""
        
        # 🔧 FIX: Guard against zero or invalid length
        if length <= 0:
            logger.warning(f"Invalid kernel length {length} - using default 1000")
            length = 1000
            
        with torch.no_grad():
            # Compute discrete system
            A_discrete = torch.exp(-self.dt * torch.abs(self.A))  # Shape: (d_state,)
            B_discrete = self.dt * self.B  # Shape: (d_state, d_model)
            
            # Generate impulse response (convolution kernel)
            kernels = []
            h = torch.zeros(self.d_state, device=self.A.device)
            
            for t in range(length):
                if t == 0:
                    # Impulse input
                    u = torch.ones(self.d_model, device=self.A.device)
                else:
                    u = torch.zeros(self.d_model, device=self.A.device)
                
                # State update
                h = h * A_discrete + torch.sum(B_discrete * u.unsqueeze(0), dim=1)
                
                # Output: y_t = C * h_t + D * u_t
                y = torch.sum(self.C * h.unsqueeze(0), dim=1) + self.D * u
                kernels.append(y)
            
            return torch.stack(kernels, dim=0)  # Shape: (length, d_model)
    
    def get_frequency_response(self, freqs, sampling_rate=250):
        """Compute frequency response for visualization"""
        with torch.no_grad():
            # Get impulse response
            kernel = self.get_kernel(length=1000)  # Shape: (1000, d_model)
            
            # Compute FFT for each output dimension
            freq_responses = []
            for dim in range(self.d_model):
                kernel_1d = kernel[:, dim].cpu().numpy()
                
                # Zero-pad for better frequency resolution
                kernel_padded = np.pad(kernel_1d, (0, 2048 - len(kernel_1d)), 'constant')
                
                # Compute FFT
                fft = np.fft.fft(kernel_padded)
                freq_response = np.abs(fft[:len(freqs)])
                freq_responses.append(freq_response)
            
            return np.array(freq_responses)  # Shape: (d_model, len(freqs))
        
    def forward(self, x):
        """Fixed S4 forward pass with proper matrix operations"""
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # 🔧 FIX: Guard against zero sequence length
        if seq_len == 0:
            logger.warning("S4 received zero sequence length - returning zeros")
            return torch.zeros(batch_size, 0, d_model, device=x.device)
        
        # Input validation
        if torch.isnan(x).any():
            logger.warning("NaNs found in S4 input - replacing with zeros")
            x = torch.zeros_like(x)
        
        # Compute discrete system with stability checks
        # A is diagonal, so exp(-dt * |A|) is element-wise
        A_discrete = torch.exp(-self.dt * torch.abs(self.A))  # Shape: (d_state,)
        B_discrete = self.dt * self.B  # Shape: (d_state, d_model)
        
        # Check for NaNs in discrete system
        if torch.isnan(A_discrete).any() or torch.isnan(B_discrete).any():
            logger.warning("NaNs in S4 discrete system - using fallback")
            return torch.zeros_like(x)
        
        # Process sequence with state space model
        h = torch.zeros(batch_size, self.d_state, device=x.device)  # Initial state
        outputs = []
        
        for t in range(seq_len):
            u_t = x[:, t, :]  # Input at time t: (batch, d_model)
            
            # State update: h_{t+1} = A_discrete * h_t + B_discrete @ u_t
            # A_discrete is diagonal, so we use element-wise multiplication
            h_next = h * A_discrete.unsqueeze(0)  # (batch, d_state) * (1, d_state)
            
            # B_discrete @ u_t: (d_state, d_model) @ (batch, d_model).T
            # Reshape for batch matrix multiplication
            B_u = torch.matmul(B_discrete, u_t.T).T  # (batch, d_state)
            
            h = h_next + B_u
            h = torch.clamp(h, min=-10, max=10)  # Stability clamp
            
            # Output: y_t = C @ h_t + D * u_t
            # C @ h: (d_model, d_state) @ (batch, d_state).T -> (d_model, batch) -> (batch, d_model)
            C_h = torch.matmul(self.C, h.T).T  # (batch, d_model)
            D_u = u_t * self.D.unsqueeze(0)  # (batch, d_model) * (1, d_model)
            
            y_t = C_h + D_u
            y_t = torch.clamp(y_t, min=-10, max=10)  # Stability clamp
            
            # NaN check
            if torch.isnan(y_t).any():
                logger.warning(f"NaNs in S4 output at step {t}")
                y_t = torch.zeros_like(y_t)
            
            outputs.append(y_t)
        
        # Stack outputs: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        output = torch.stack(outputs, dim=1)
        
        # Final safety check
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
        
        # Store downsample factor for conditional downsampling
        self.downsample_factor = downsample_factor
        
        # 1. Conditional input downsampling for efficiency (reduces sequence length)
        if downsample_factor > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=downsample_factor, padding=1),
                nn.BatchNorm1d(input_dim),  # Use BatchNorm1d instead of LayerNorm for conv output
                nn.ReLU()
            )
            print(f"  🔧 Using {downsample_factor}x downsampling for efficiency")
        else:
            self.downsample = nn.Identity()
            print(f"  🔧 No downsampling - preserving raw temporal resolution")
        
        # 2. Input projection to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 3. Conservative positional encoding (smaller for downsampled sequences)
        max_downsampled_len = max_seq_len // downsample_factor
        self.positional_encoding = nn.Parameter(torch.randn(1, max_downsampled_len, d_model) * 0.0001)
        
        # 4. Single optimized S4 layer (reduced from multiple layers)
        self.s4_layers = nn.ModuleList([
            OptimizedS4Layer(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        
        # 5. Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # 6. Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Store parameters for logging
        self.downsample_factor = downsample_factor
        
        logger.info(f"OptimizedS4Classifier: d_model={d_model}, d_state={d_state}, n_layers={n_layers}")
        if downsample_factor > 1:
            logger.info(f"  Expected speedup: ~{downsample_factor**2}x from downsampling + state reduction")
        else:
            logger.info(f"  No downsampling - optimizing for long-range modeling over efficiency")
        
    def forward(self, x):
        # Input validation
        if torch.isnan(x).any():
            logger.warning("NaNs in OptimizedS4Classifier input")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)
        
        # 🔧 FIX: Guard against zero sequence length
        batch_size, channels, seq_len = x.shape
        if seq_len == 0:
            logger.warning("S4Classifier received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)

        # x shape: (batch, channels, time)
        
        # 1. Downsample input to reduce sequence length
        x_downsampled = self.downsample(x)  # (batch, channels, time//downsample_factor)
        
        # 🔧 FIX: Check downsampled sequence isn't empty
        if x_downsampled.size(-1) == 0:
            logger.warning("S4Classifier downsampling resulted in zero length - using smaller downsample")
            # Fallback: reduce downsample factor if sequence becomes too short
            fallback_downsample = max(1, seq_len // 10)  # At least 10 samples after downsampling
            x_downsampled = F.avg_pool1d(x, kernel_size=fallback_downsample, stride=fallback_downsample)
        
        # 2. Transpose for S4: (batch, channels, time) -> (batch, time, channels)
        x = x_downsampled.transpose(1, 2)
        
        # 3. Project to model dimension
        x = self.input_projection(x)
        
        if torch.isnan(x).any():
            logger.warning("NaNs after input projection")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)
        
        # 4. Add positional encoding (now shorter due to downsampling)
        seq_len = x.size(1)
        
        # 🔧 FIX: Guard against zero sequence length after downsampling
        if seq_len == 0:
            logger.warning("S4Classifier sequence length is zero after downsampling")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)
        
        # 🔧 FIX: Safely handle positional encoding with proper extension
        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :]
        else:
            # Handle edge case where sequence is longer than expected
            repeat_factor = (seq_len // self.positional_encoding.size(1)) + 1
            extended_pos = self.positional_encoding.repeat(1, repeat_factor, 1)
            pos_enc = extended_pos[:, :seq_len, :]
            
        x = x + pos_enc
        
        # 5. Apply S4 layers with residual connections
        for i, layer in enumerate(self.s4_layers):
            residual = x
            x_new = layer(x)
            
            # Safe residual connection with nonlinearity
            if torch.isnan(x_new).any():
                logger.warning(f"NaNs from OptimizedS4 layer {i+1}, using residual only")
                x = residual
            else:
                # Add nonlinearity and layer norm for better learning
                x = self.layer_norms[i](F.relu(residual + x_new))
        
        # 6. Global pooling and classification
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        if torch.isnan(x).any():
            logger.warning("NaNs after global pooling")
            x = torch.zeros_like(x)
        
        output = self.classifier(x)
        
        # Final safety check
        if torch.isnan(output).any():
            logger.warning("NaNs in final OptimizedS4 output")
            output = torch.zeros_like(output)
        
        return output

# Keep the original S4Classifier as S4Classifier for backward compatibility
S4Classifier = OptimizedS4Classifier

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
        
        # 🔧 FIX: Guard against mismatched dimensions
        batch_size, seq_len, input_dim = x.shape
        
        if seq_len == 0:
            logger.warning("Transformer received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)
        
        # Project and add positional encoding with proper bounds checking
        x = self.input_projection(x)
        
        # 🔧 FIX: Safely slice positional encoding to match input length
        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed using repeat
            repeat_factor = (seq_len // self.positional_encoding.size(1)) + 1
            extended_pos = self.positional_encoding.repeat(1, repeat_factor, 1)
            pos_enc = extended_pos[:, :seq_len, :]
            
        x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling and classification
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        return self.classifier(x)

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
        x = x.squeeze(-1)  # Remove last dimension
        return self.classifier(x)

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
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(x)
        
        # Use last output
        x = output[:, -1, :]  # (batch, hidden_dim * 2)
        
        return self.classifier(x)

class S5Classifier(nn.Module):
    """S5-based EEG Classifier - Next Generation State Space Model"""
    
    def __init__(self, input_dim=64, hidden_dim=128, n_layers=2, n_classes=3, dropout=0.1):
        super().__init__()
        
        # Import S5 here to avoid import errors if package not available
        try:
            from s5 import S5Block
        except ImportError:
            raise ImportError("s5-pytorch not installed. Install with: pip install s5-pytorch")
        
        # Input projection to prepare for S5
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack of S5 blocks for temporal modeling
        self.s5_layers = nn.ModuleList([
            S5Block(
                dim=hidden_dim,
                state_dim=64,  # State dimension for S5
                bidir=True,   # 🔧 CRITICAL: Use bidirectional for better performance
                ff_dropout=dropout,
                attn_dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        logger.info(f"S5Classifier: hidden_dim={hidden_dim}, n_layers={n_layers}, state_dim=64")
        
    def forward(self, x):
        # Input validation
        if torch.isnan(x).any():
            logger.warning("NaNs in S5Classifier input")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)
        
        # Guard against zero sequence length
        batch_size, channels, seq_len = x.shape
        if seq_len == 0:
            logger.warning("S5Classifier received zero sequence length - returning zeros")
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=x.device)
        
        # x shape: (batch, channels, time) -> (batch, time, channels) for S5
        x = x.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (batch, time, hidden_dim)
        
        # Apply S5 layers with residual connections
        for i, (s5_layer, norm) in enumerate(zip(self.s5_layers, self.layer_norms)):
            residual = x
            
            try:
                # S5 forward pass
                x_new = s5_layer(x)  # S5Block handles [B, L, D] format
                
                # Residual connection and layer norm
                x = norm(x_new + residual)
                
            except Exception as e:
                logger.warning(f"S5 layer {i+1} failed: {e}, using residual")
                x = residual
        
        # Global pooling: (batch, time, hidden_dim) -> (batch, hidden_dim, time) -> (batch, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, time)
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_dim)
        
        # Classification
        if torch.isnan(x).any():
            logger.warning("NaNs in S5 output before classification")
            return torch.zeros(x.size(0), self.classifier[-1].out_features, device=x.device)
        
        output = self.classifier(x)
        
        # Final safety check
        if torch.isnan(output).any():
            logger.warning("NaNs in S5 final output")
            return torch.zeros_like(output)
        
        return output

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class SimpleHBNManager:
    """Simple HBN data manager for loading EEG data"""
    
    def __init__(self, base_path="/home/ergezerm/hbn_eeg"):
        self.base_path = Path(base_path)
        
    def get_available_subjects(self, release=1):
        """Get list of available subjects for a release"""
        if release == 1:
            release_path = self.base_path / f"release_{release}" / "ds005505"
        elif release == 4:
            release_path = self.base_path / f"release_{release}" / "ds005508"
        else:
            logger.warning(f"Unsupported release: {release}")
            return []
            
        if not release_path.exists():
            logger.warning(f"Release path not found: {release_path}")
            return []
        
        subjects = []
        for sub_dir in release_path.iterdir():
            if sub_dir.is_dir() and sub_dir.name.startswith('sub-'):
                subjects.append(sub_dir.name)
        
        return sorted(subjects)
    
    def load_subject_data(self, subject_id, release=1, task="DespicableMe"):
        """Load EEG data for a specific subject and task"""
        if release == 1:
            release_path = self.base_path / f"release_{release}" / "ds005505"
        elif release == 4:
            release_path = self.base_path / f"release_{release}" / "ds005508"
        else:
            logger.warning(f"Unsupported release: {release}")
            return None
            
        subject_path = release_path / subject_id / "eeg"
        
        # Find the task file
        eeg_file = subject_path / f"{subject_id}_task-{task}_eeg.set"
        
        if not eeg_file.exists():
            return None
            
        try:
            # Load EEGLAB format
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
            
            # Basic preprocessing
            raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)  # Bandpass filter
            raw.set_eeg_reference('average', verbose=False)  # Average reference
            
            # Get data
            data = raw.get_data()  # Shape: (n_channels, n_timepoints)
            sfreq = raw.info['sfreq']
            
            return {
                'data': data,
                'sfreq': sfreq,
                'ch_names': raw.ch_names,
                'duration': data.shape[1] / sfreq
            }
            
        except Exception as e:
            logger.warning(f"Failed to load {subject_id} {task}: {e}")
            return None

class ICSSPMovieSegmentDataset(Dataset):
    """Dataset for ICASSP movie segment classification benchmarking"""
    
    def __init__(self, subjects, manager, segment_length_sec=8, overlap_ratio=0.5,
                 max_subjects=30, target_channels=64, label_mode='movie', release=1, max_movie_classes=None):
        
        self.segment_length_sec = segment_length_sec
        self.overlap_ratio = overlap_ratio
        self.target_channels = target_channels
        self.label_mode = label_mode  # 'movie', 'subject', or 'movie+subject'
        self.release = release
        self.max_movie_classes = max_movie_classes
        
        # Movie tasks for classification - EXPANDED for better differentiation
        # Order by expected data availability (common tasks first)
        all_movie_tasks = ["DespicableMe", "DiaryOfAWimpyKid", "ThePresent", "FunWithFractals", "RestingState", "TeenageMutantNinjaTurtles", "SimonGame"]
        
        # Limit movie tasks if specified
        if max_movie_classes is not None:
            self.movie_tasks = all_movie_tasks[:max_movie_classes]
        else:
            self.movie_tasks = all_movie_tasks[:5]  # Default to 5 classes
        
        logger.info(f"Creating ICASSP Movie Dataset...")
        logger.info(f"   Release: {release}")
        logger.info(f"   Segment length: {segment_length_sec}s")
        logger.info(f"   Overlap ratio: {overlap_ratio}")
        logger.info(f"   Max subjects: {max_subjects}")
        logger.info(f"   Target channels: {target_channels}")
        logger.info(f"   Label mode: {label_mode}")
        
        # Create subject mapping for subject classification
        subjects_used = subjects[:max_subjects]
        self.subject_to_idx = {subj: idx for idx, subj in enumerate(subjects_used)}
        
        # Load and process data
        self.segments = []
        self.labels = []
        self.metadata = []
        self.movie_labels = []  # Track which movie each segment belongs to
        
        logger.info(f"   Processing {len(subjects_used)} subjects...")
        
        successful_movies = []  # Track which movies actually have data
        movie_segment_counts = {movie: 0 for movie in self.movie_tasks}
        
        for i, subject_id in enumerate(subjects_used):
            logger.info(f"     {i+1}/{len(subjects_used)}: {subject_id}")
            
            for movie_idx, movie in enumerate(self.movie_tasks):
                data = manager.load_subject_data(subject_id, release=release, task=movie)
                
                if data is not None:
                    segments, seg_labels, seg_meta = self._extract_segments(
                        data, movie_idx, subject_id, movie, i
                    )
                    
                    if segments:  # Only add if we got segments
                        self.segments.extend(segments)
                        self.labels.extend(seg_labels)
                        self.metadata.extend(seg_meta)
                        # Track movie labels
                        self.movie_labels.extend([movie] * len(segments))
                        movie_segment_counts[movie] += len(segments)
                        
                        if movie not in successful_movies:
                            successful_movies.append(movie)
        
        # Filter out movies with no segments
        self.successful_movies = [movie for movie in self.movie_tasks if movie_segment_counts[movie] > 0]
        
        # Convert to arrays
        self.segments = np.array(self.segments) if self.segments else np.array([])
        self.labels = np.array(self.labels) if self.labels else np.array([])
        
        # Remap labels to be consecutive (remove gaps from missing movies)
        if len(self.labels) > 0:
            unique_labels = np.unique(self.labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.labels = np.array([label_mapping[label] for label in self.labels])
            
            # Update movie_labels mapping
            movie_to_new_idx = {}
            for movie, old_idx in zip(self.movie_tasks, range(len(self.movie_tasks))):
                if old_idx in label_mapping:
                    movie_to_new_idx[movie] = label_mapping[old_idx]
            
            # Update successful movies list to match new indices
            self.unique_movies = []
            for movie in self.movie_tasks:
                if movie in movie_to_new_idx:
                    self.unique_movies.append(movie)
        else:
            self.unique_movies = []
        
        # Dynamic class counting
        self.n_classes = len(np.unique(self.labels)) if len(self.labels) > 0 else 0
        
        logger.info(f"Dataset created:")
        logger.info(f"   Total segments: {len(self.segments)}")
        logger.info(f"   Classes: {self.n_classes}")
        logger.info(f"   Available movies: {self.unique_movies}")
        if len(self.segments) > 0:
            logger.info(f"   Segment shape: {self.segments[0].shape}")
            logger.info(f"   Class distribution: {np.bincount(self.labels)}")
            
            # Show movie distribution
            from collections import Counter
            movie_counts = Counter(self.movie_labels)
            logger.info(f"   Movie distribution:")
            for movie, count in movie_counts.most_common():
                logger.info(f"     {movie}: {count} segments")
    
    def _extract_segments(self, data, movie_idx, subject_id, movie, subject_idx):
        """Extract overlapping segments from continuous EEG data"""
        eeg_data = data['data']  # Shape: (n_channels, n_timepoints)
        sfreq = data['sfreq']
        
        # Downsample to 250 Hz if needed
        if sfreq > 250:
            from scipy import signal
            downsample_factor = int(sfreq // 250)
            eeg_data = signal.decimate(eeg_data, downsample_factor, axis=1)
            sfreq = sfreq / downsample_factor
        
        # Select channels (take first N channels, or pad if needed)
        if eeg_data.shape[0] > self.target_channels:
            eeg_data = eeg_data[:self.target_channels]
        elif eeg_data.shape[0] < self.target_channels:
            # Pad with zeros if fewer channels
            pad_channels = self.target_channels - eeg_data.shape[0]
            padding = np.zeros((pad_channels, eeg_data.shape[1]))
            eeg_data = np.vstack([eeg_data, padding])
        
        # Extract segments
        segment_samples = int(self.segment_length_sec * sfreq)
        step_samples = int(segment_samples * (1 - self.overlap_ratio))
        
        segments = []
        labels = []
        metadata = []
        
        for start_idx in range(0, eeg_data.shape[1] - segment_samples + 1, step_samples):
            end_idx = start_idx + segment_samples
            segment = eeg_data[:, start_idx:end_idx]  # Shape: (channels, samples)
            
            # Normalize segment
            segment = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-8)
            
            segments.append(segment)
            
            # Dynamic labeling based on mode
            if self.label_mode == 'movie':
                label = movie_idx
            elif self.label_mode == 'subject':
                label = subject_idx
            elif self.label_mode == 'movie+subject':
                # Combine movie and subject for multi-task classification
                label = movie_idx * len(self.subject_to_idx) + subject_idx
            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}")
            
            labels.append(label)
            metadata.append({
                'subject': subject_id,
                'subject_idx': subject_idx,
                'movie': movie,
                'movie_idx': movie_idx,
                'start_time': start_idx / sfreq,
                'end_time': end_idx / sfreq,
                'label': label
            })
        
        return segments, labels, metadata
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.segments[idx]), torch.LongTensor([self.labels[idx]])[0]
    
    def balance_classes(self, min_samples_per_class=100, max_samples_per_class=None):
        """Balance classes by removing classes with too few samples and optionally limiting max samples"""
        
        if len(self.labels) == 0:
            logger.warning("No samples to balance")
            return
        
        # Count samples per class
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        logger.info(f"Before balancing: {dict(zip(unique_labels, counts))}")
        
        # Filter out classes with too few samples
        valid_classes = unique_labels[counts >= min_samples_per_class]
        
        if len(valid_classes) == 0:
            logger.error(f"No classes have at least {min_samples_per_class} samples!")
            return
        
        # Keep only valid classes
        valid_indices = []
        valid_labels = []
        valid_movie_labels = []
        valid_metadata = []
        
        for i, label in enumerate(self.labels):
            if label in valid_classes:
                valid_indices.append(i)
                valid_labels.append(label)
                valid_movie_labels.append(self.movie_labels[i])
                valid_metadata.append(self.metadata[i])
        
        # Update datasets
        self.segments = self.segments[valid_indices]
        self.labels = np.array(valid_labels)
        self.movie_labels = valid_movie_labels
        self.metadata = valid_metadata
        
        # Remap labels to be consecutive
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        
        # Update unique movies
        remaining_movies = []
        for movie in self.unique_movies:
            if any(meta['movie'] == movie for meta in self.metadata):
                remaining_movies.append(movie)
        self.unique_movies = remaining_movies
        
        # Update class count
        self.n_classes = len(valid_classes)
        
        # Optionally limit max samples per class
        if max_samples_per_class is not None:
            self._limit_samples_per_class(max_samples_per_class)
        
        # Final counts
        final_counts = np.bincount(self.labels)
        logger.info(f"After balancing: {final_counts}")
        logger.info(f"Remaining movies: {self.unique_movies}")
    
    def _limit_samples_per_class(self, max_samples):
        """Limit the number of samples per class"""
        
        indices_to_keep = []
        
        for class_idx in range(self.n_classes):
            class_indices = np.where(self.labels == class_idx)[0]
            
            if len(class_indices) > max_samples:
                # Randomly sample max_samples
                np.random.shuffle(class_indices)
                class_indices = class_indices[:max_samples]
            
            indices_to_keep.extend(class_indices)
        
        # Sort indices to maintain order
        indices_to_keep = sorted(indices_to_keep)
        
        # Update all arrays
        self.segments = self.segments[indices_to_keep]
        self.labels = self.labels[indices_to_keep]
        self.movie_labels = [self.movie_labels[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]

class ICSSPMovieSegmentDatasetMultiRelease(Dataset):
    """Multi-release dataset for ICASSP movie segment classification with enhanced data coverage"""
    
    def __init__(self, subjects_with_releases, manager, segment_length_sec=8, overlap_ratio=0.5,
                 target_channels=64, label_mode='movie', max_movie_classes=None):
        
        self.segment_length_sec = segment_length_sec
        self.overlap_ratio = overlap_ratio
        self.target_channels = target_channels
        self.label_mode = label_mode
        self.max_movie_classes = max_movie_classes
        
        # Movie tasks for classification
        all_movie_tasks = ["DespicableMe", "DiaryOfAWimpyKid", "ThePresent", "FunWithFractals", "RestingState", "TeenageMutantNinjaTurtles", "SimonGame"]
        
        if max_movie_classes is not None:
            self.movie_tasks = all_movie_tasks[:max_movie_classes]
        else:
            self.movie_tasks = all_movie_tasks[:5]
        
        logger.info(f"Creating Multi-Release ICASSP Movie Dataset...")
        logger.info(f"   Segment length: {segment_length_sec}s")
        logger.info(f"   Overlap ratio: {overlap_ratio}")
        logger.info(f"   Subjects from multiple releases: {len(subjects_with_releases)}")
        logger.info(f"   Target channels: {target_channels}")
        logger.info(f"   Label mode: {label_mode}")
        
        # Create subject mapping
        self.subject_to_idx = {f"{subj}_{release}": idx for idx, (subj, release) in enumerate(subjects_with_releases)}
        
        # Load and process data
        self.segments = []
        self.labels = []
        self.metadata = []
        self.movie_labels = []
        
        logger.info(f"   Processing {len(subjects_with_releases)} subjects...")
        
        movie_segment_counts = {movie: 0 for movie in self.movie_tasks}
        
        for i, (subject_id, release) in enumerate(subjects_with_releases):
            if (i + 1) % 10 == 0:
                logger.info(f"     {i+1}/{len(subjects_with_releases)}: {subject_id} (Release {release})")
            
            for movie_idx, movie in enumerate(self.movie_tasks):
                data = manager.load_subject_data(subject_id, release=release, task=movie)
                
                if data is not None:
                    segments, seg_labels, seg_meta = self._extract_segments(
                        data, movie_idx, subject_id, movie, i, release
                    )
                    
                    if segments:
                        self.segments.extend(segments)
                        self.labels.extend(seg_labels)
                        self.metadata.extend(seg_meta)
                        self.movie_labels.extend([movie] * len(segments))
                        movie_segment_counts[movie] += len(segments)
        
        # Convert to arrays
        self.segments = np.array(self.segments) if self.segments else np.array([])
        self.labels = np.array(self.labels) if self.labels else np.array([])
        
        # Remap labels to be consecutive
        if len(self.labels) > 0:
            unique_labels = np.unique(self.labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.labels = np.array([label_mapping[label] for label in self.labels])
            
            # Update movie lists
            self.unique_movies = []
            for movie in self.movie_tasks:
                if movie in [meta['movie'] for meta in self.metadata]:
                    self.unique_movies.append(movie)
        else:
            self.unique_movies = []
        
        # Dynamic class counting
        self.n_classes = len(np.unique(self.labels)) if len(self.labels) > 0 else 0
        
        logger.info(f"Multi-Release Dataset created:")
        logger.info(f"   Total segments: {len(self.segments)}")
        logger.info(f"   Classes: {self.n_classes}")
        logger.info(f"   Available movies: {self.unique_movies}")
        if len(self.segments) > 0:
            logger.info(f"   Segment shape: {self.segments[0].shape}")
            logger.info(f"   Class distribution: {np.bincount(self.labels)}")
            
            # Show movie distribution
            from collections import Counter
            movie_counts = Counter(self.movie_labels)
            logger.info(f"   Movie distribution:")
            for movie, count in movie_counts.most_common():
                logger.info(f"     {movie}: {count} segments")
    
    def _extract_segments(self, data, movie_idx, subject_id, movie, subject_idx, release):
        """Extract overlapping segments from continuous EEG data"""
        eeg_data = data['data']
        sfreq = data['sfreq']
        
        # Downsample to 250 Hz if needed
        if sfreq > 250:
            from scipy import signal
            downsample_factor = int(sfreq // 250)
            eeg_data = signal.decimate(eeg_data, downsample_factor, axis=1)
            sfreq = sfreq / downsample_factor
        
        # Select channels
        if eeg_data.shape[0] > self.target_channels:
            eeg_data = eeg_data[:self.target_channels]
        elif eeg_data.shape[0] < self.target_channels:
            pad_channels = self.target_channels - eeg_data.shape[0]
            padding = np.zeros((pad_channels, eeg_data.shape[1]))
            eeg_data = np.vstack([eeg_data, padding])
        
        # Extract segments
        segment_samples = int(self.segment_length_sec * sfreq)
        step_samples = int(segment_samples * (1 - self.overlap_ratio))
        
        segments = []
        labels = []
        metadata = []
        
        for start_idx in range(0, eeg_data.shape[1] - segment_samples + 1, step_samples):
            end_idx = start_idx + segment_samples
            segment = eeg_data[:, start_idx:end_idx]
            
            # Normalize segment
            segment = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-8)
            
            segments.append(segment)
            
            # Dynamic labeling
            if self.label_mode == 'movie':
                label = movie_idx
            elif self.label_mode == 'subject':
                label = subject_idx
            elif self.label_mode == 'movie+subject':
                label = movie_idx * len(self.subject_to_idx) + subject_idx
            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}")
            
            labels.append(label)
            metadata.append({
                'subject': subject_id,
                'subject_idx': subject_idx,
                'release': release,
                'movie': movie,
                'movie_idx': movie_idx,
                'start_time': start_idx / sfreq,
                'end_time': end_idx / sfreq,
                'label': label
            })
        
        return segments, labels, metadata
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.segments[idx]), torch.LongTensor([self.labels[idx]])[0]
    
    def balance_classes(self, min_samples_per_class=50, max_samples_per_class=None):
        """Balance classes with more lenient requirements for longer sequences"""
        
        if len(self.labels) == 0:
            logger.warning("No samples to balance")
            return
        
        # Count samples per class
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        logger.info(f"Before balancing: {dict(zip(unique_labels, counts))}")
        
        # Filter out classes with too few samples
        valid_classes = unique_labels[counts >= min_samples_per_class]
        
        if len(valid_classes) == 0:
            logger.error(f"No classes have at least {min_samples_per_class} samples!")
            return
        
        # Keep only valid classes
        valid_indices = []
        valid_labels = []
        valid_movie_labels = []
        valid_metadata = []
        
        for i, label in enumerate(self.labels):
            if label in valid_classes:
                valid_indices.append(i)
                valid_labels.append(label)
                valid_movie_labels.append(self.movie_labels[i])
                valid_metadata.append(self.metadata[i])
        
        # Update datasets
        self.segments = self.segments[valid_indices]
        self.labels = np.array(valid_labels)
        self.movie_labels = valid_movie_labels
        self.metadata = valid_metadata
        
        # Remap labels to be consecutive
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        
        # Update unique movies
        remaining_movies = []
        for movie in self.unique_movies:
            if any(meta['movie'] == movie for meta in self.metadata):
                remaining_movies.append(movie)
        self.unique_movies = remaining_movies
        
        # Update class count
        self.n_classes = len(valid_classes)
        
        # Optionally limit max samples per class
        if max_samples_per_class is not None:
            self._limit_samples_per_class(max_samples_per_class)
        
        # Final counts
        final_counts = np.bincount(self.labels)
        logger.info(f"After balancing: {final_counts}")
        logger.info(f"Remaining movies: {self.unique_movies}")
    
    def _limit_samples_per_class(self, max_samples):
        """Limit the number of samples per class"""
        
        indices_to_keep = []
        
        for class_idx in range(self.n_classes):
            class_indices = np.where(self.labels == class_idx)[0]
            
            if len(class_indices) > max_samples:
                # Randomly sample max_samples
                np.random.shuffle(class_indices)
                class_indices = class_indices[:max_samples]
            
            indices_to_keep.extend(class_indices)
        
        # Sort indices to maintain order
        indices_to_keep = sorted(indices_to_keep)
        
        # Update all arrays
        self.segments = self.segments[indices_to_keep]
        self.labels = self.labels[indices_to_keep]
        self.movie_labels = [self.movie_labels[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class ModelTrainer:
    """Comprehensive model trainer and evaluator with advanced features"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 use_ema=True, grad_clip_norm=1.0, log_interval=5):
        self.device = device
        self.results = {}
        self.use_ema = use_ema and EMA_AVAILABLE
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        if self.use_ema:
            logger.info("EMA (Exponential Moving Average) enabled")
        else:
            logger.info("EMA not available or disabled")
    
    def train_model(self, model, train_loader, val_loader, model_name,
                   n_epochs=50, lr=1e-3, patience=10):
        """Train a single model with advanced features"""
        
        logger.info(f"Training {model_name}...")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        model = model.to(self.device)
        
        # Optimized training setup
        if model_name in ['S4', 'OptimizedS4']:
            # Optimized settings for S4 models
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # Higher weight decay
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)  # Cosine scheduler
            logger.info(f"  Using optimized AdamW + CosineAnnealingLR for {model_name}")
        else:
            # Standard settings for other models
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 🔍 DIAGNOSTIC: Calculate class-balanced loss weights
        try:
            # Collect all training labels to calculate class weights
            all_train_labels = []
            for _, y_batch in train_loader:
                all_train_labels.extend(y_batch.cpu().numpy())
            
            from sklearn.utils.class_weight import compute_class_weight
            unique_classes = np.unique(all_train_labels)
            class_weights = compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=all_train_labels
            )
            
            # Convert to tensor
            weight_tensor = torch.zeros(len(unique_classes))
            for i, class_id in enumerate(unique_classes):
                weight_tensor[class_id] = class_weights[i]
            
            criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))
            
            logger.info(f"  🔍 Using class-balanced loss weights: {dict(zip(unique_classes, class_weights))}")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to calculate class weights: {e}")
            criterion = nn.CrossEntropyLoss()
            logger.info(f"  Using standard CrossEntropyLoss")
        
        # Setup EMA if available
        ema = None
        if self.use_ema:
            # Use configurable EMA decay rate
            ema_decay = 0.98 if hasattr(self, 'ema_decay') else 0.995
            ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        
        best_val_acc = 0
        best_val_loss = float('inf')  # Track validation loss too
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        val_losses = []  # Track validation losses
        learning_rates = []
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 🔧 FIX: Add defensive input shape validation
                batch_size, channels, seq_len = X_batch.shape
                if seq_len == 0:
                    logger.warning(f"Skipping batch with zero sequence length: {X_batch.shape}")
                    continue
                if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                    logger.warning(f"Skipping batch with NaNs/Infs: {X_batch.shape}")
                    continue
                
                # Log input shape for debugging (only occasionally)
                if n_batches == 0:
                    logger.info(f"Model {model_name}: Input shape = {X_batch.shape}")
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(X_batch)
                except Exception as e:
                    logger.error(f"Model {model_name} failed on input shape {X_batch.shape}: {str(e)}")
                    raise e  # Re-raise to be caught by outer try-catch
                    
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Enhanced gradient clipping for S4 stability
                if model_name in ['S4', 'OptimizedS4']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Aggressive clipping for S4
                elif self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
                
                optimizer.step()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Check for NaN and handle gracefully
            if torch.isnan(torch.tensor(avg_loss)):
                logger.warning(f"  NaN loss detected at epoch {epoch}. Stopping training for {model_name}")
                break
            
            # Validation (use EMA model if available)
            if ema is not None:
                with ema.average_parameters():
                    val_acc, val_loss = self.evaluate_model_with_loss(model, val_loader, criterion)
            else:
                val_acc, val_loss = self.evaluate_model_with_loss(model, val_loader, criterion)
            
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            # Update scheduler based on type
            if model_name in ['S4', 'OptimizedS4']:
                scheduler.step()  # Cosine annealing steps every epoch
            else:
                scheduler.step(val_loss)  # ReduceLROnPlateau uses validation loss
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model (with EMA if available)
                if ema is not None:
                    with ema.average_parameters():
                        torch.save(model.state_dict(), f'results/best_{model_name.lower()}_model.pth')
                else:
                    torch.save(model.state_dict(), f'results/best_{model_name.lower()}_model.pth')
            else:
                patience_counter += 1
            
            # Enhanced logging with loss information
            if epoch % self.log_interval == 0 or epoch == n_epochs - 1:
                logger.info(f"  Epoch {epoch:3d}/{n_epochs}: "
                           f"Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, "
                           f"Val Acc={val_acc:.3f}, LR={learning_rates[-1]:.2e}, "
                           f"Best Acc={best_val_acc:.3f}")
                
                # Log first batch outputs for S4 debugging
                if model_name == 'S4' and epoch % (self.log_interval * 2) == 0:
                    with torch.no_grad():
                        sample_batch = next(iter(val_loader))
                        X_sample, y_sample = sample_batch[0][:2].to(self.device), sample_batch[1][:2].to(self.device)
                        sample_outputs = model(X_sample)
                        sample_probs = torch.softmax(sample_outputs, dim=-1)
                        logger.info(f"  S4 Sample Outputs: {sample_outputs[0].cpu().numpy()}")
                        logger.info(f"  S4 Sample Probs: {sample_probs[0].cpu().numpy()}")
                        logger.info(f"  S4 Sample Targets: {y_sample[:2].cpu().numpy()}")
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'results/best_{model_name.lower()}_model.pth'))
        
        # Generate learning curves
        self._plot_learning_curves(train_losses, val_accuracies, val_losses, learning_rates, model_name)
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'training_time': total_time,
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'ema_used': ema is not None
        }
    
    def _plot_learning_curves(self, train_losses, val_accuracies, val_losses, learning_rates, model_name):
        """Generate and save learning curve plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        # Training loss
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{model_name} - Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Validation loss
        axes[0, 1].plot(val_losses, label='Val Loss', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title(f'{model_name} - Validation Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Validation accuracy
        axes[1, 0].plot(val_accuracies, label='Val Accuracy', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title(f'{model_name} - Validation Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot(learning_rates, label='Learning Rate', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title(f'{model_name} - Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower()}_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Learning curves saved to plots/{model_name.lower()}_learning_curves.png")
    
    def train_model_old(self, model, train_loader, val_loader, model_name,
                   n_epochs=50, lr=1e-3, patience=10):
        """Train a single model (original version for reference)"""
        
        logger.info(f"Training {model_name} (original)...")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        model = model.to(self.device)
        
        # Setup training
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        val_losses = []  # Track validation losses
        learning_rates = []
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping for training stability
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch = X_val_batch.to(self.device)
                    y_val_batch = y_val_batch.to(self.device)
                    
                    val_outputs = model(X_val_batch)
                    val_loss_batch = criterion(val_outputs, y_val_batch)
                    val_loss += val_loss_batch.item()
                    
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += y_val_batch.size(0)
                    correct += (predicted == y_val_batch).sum().item()
            
            val_acc = correct / total
            val_losses.append(val_loss / len(val_loader))
            
            scheduler.step(val_loss)  # Use validation loss for scheduling
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'results/best_{model_name.lower()}_model.pth')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % self.log_interval == 0 or epoch == n_epochs - 1:
                logger.info(f"  Epoch {epoch:3d}/{n_epochs}: "
                           f"Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, "
                           f"Val Acc={val_acc:.3f}, LR={learning_rates[-1]:.2e}, "
                           f"Best Acc={best_val_acc:.3f}")
        
        total_time = time.time() - start_time
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'results/best_{model_name.lower()}_model.pth'))
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'training_time': total_time,
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates
        }
    
    def evaluate_model(self, model, data_loader):
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        return correct / total
    
    def evaluate_model_with_loss(self, model, data_loader, criterion):
        """Evaluate model accuracy and loss"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                total_loss += loss.item()
                n_batches += 1
        
        return correct / total, total_loss / n_batches
    
    def comprehensive_evaluation(self, model, test_loader, model_name):
        """Comprehensive model evaluation with memory usage"""
        
        logger.info(f"Comprehensive evaluation of {model_name}...")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # 🔍 DIAGNOSTIC: Collect probabilities for entropy analysis
        inference_times = []
        
        # Parameter count
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 🔧 FIX: Add defensive input validation in evaluation
                if X_batch.shape[-1] == 0:
                    logger.warning(f"Skipping evaluation batch with zero sequence length: {X_batch.shape}")
                    continue
                
                # Measure inference time
                start_time = time.time()
                
                try:
                    outputs = model(X_batch)
                except Exception as e:
                    logger.error(f"Model {model_name} failed during evaluation on input shape {X_batch.shape}: {str(e)}")
                    raise e
                    
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(X_batch))  # Per sample
                
                # 🔍 DIAGNOSTIC: Collect probabilities for entropy analysis
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Get peak memory usage
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # 🔍 DIAGNOSTIC: Class Distribution Analysis
        from collections import Counter
        import torch.nn.functional as F
        
        pred_counts = Counter(all_preds)
        true_counts = Counter(all_labels)
        
        logger.info(f"🔍 {model_name} Class Distribution Analysis:")
        logger.info(f"  True label counts: {dict(sorted(true_counts.items()))}")
        logger.info(f"  Predicted label counts: {dict(sorted(pred_counts.items()))}")
        
        # Check for missing classes in predictions
        missing_classes = set(true_counts.keys()) - set(pred_counts.keys())
        if missing_classes:
            logger.warning(f"  ⚠️  Classes NEVER predicted: {missing_classes}")
            logger.warning(f"  ⚠️  This will cause precision warnings and low F1 scores!")
        
        # Check class balance
        total_samples = len(all_labels)
        logger.info(f"  True class balance:")
        for class_id, count in sorted(true_counts.items()):
            percentage = (count / total_samples) * 100
            logger.info(f"    Class {class_id}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Calculate metrics with different averaging methods
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        logger.info(f"  F1 Scores - Weighted: {f1_weighted:.4f}, Macro: {f1_macro:.4f}, Micro: {f1_micro:.4f}")
        
        # Check for mode collapse (low prediction entropy)
        if len(all_probs) > 0:
            probs_tensor = torch.tensor(all_probs)
            entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + 1e-9), dim=1).mean().item()
            logger.info(f"  Mean prediction entropy: {entropy:.4f} (>1.0 is good, <0.5 suggests collapse)")
            
            if entropy < 0.5:
                logger.warning(f"  ⚠️  Low entropy ({entropy:.4f}) suggests mode collapse!")
        
        results = {
            'accuracy': accuracy * 100,  # Percentage
            'f1_score': f1_weighted,  # Keep weighted for compatibility
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'prediction_entropy': entropy if len(all_probs) > 0 else 0,
            'missing_classes': list(missing_classes),
            'parameters': n_params,
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'peak_memory_mb': peak_memory_mb,
            'classification_report': classification_report(all_labels, all_preds, zero_division=0)
        }
        
        return results

def export_results_to_csv(all_results, config, timestamp):
    """Export results to CSV for easy table generation"""
    
    # Create results DataFrame
    data = []
    for model_name, results in all_results.items():
        data.append({
            'Model': model_name,
            'Accuracy (%)': f"{results['accuracy']:.1f}",
            'F1-Score': f"{results['f1_score']:.3f}",
            'F1-Macro': f"{results.get('f1_macro', 0):.3f}",
            'F1-Micro': f"{results.get('f1_micro', 0):.3f}",
            'Prediction Entropy': f"{results.get('prediction_entropy', 0):.3f}",
            'Missing Classes': str(results.get('missing_classes', [])),
            'Parameters': f"{results['parameters']:,}",
            'Training Time (s)': f"{results['training_time']:.1f}",
            'Inference Time (ms)': f"{results['avg_inference_time_ms']:.1f}",
            'Peak Memory (MB)': f"{results['peak_memory_mb']:.1f}",
            'Epochs Trained': results['epochs_trained'],
            'EMA Used': results.get('ema_used', False)
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_file = f"results/icassp_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Results exported to {csv_file}")
    
    return csv_file

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline with all improvements"""
    
    print("🔁 Entered main() function")
    
    logger.info("="*80)
    logger.info("ICASSP EEG Benchmarking Pipeline Starting")
    logger.info("="*80)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    logger.info("Anomaly detection enabled for debugging")
    
    # Configuration
    config = {
        'release': 1,  # Can be 1 or 4
        'max_subjects': 40,  # Use more subjects for overnight training
        'segment_length_sec': 8,
        'overlap_ratio': 0.5,
        'target_channels': 64,
        'label_mode': 'movie',  # 'movie', 'subject', or 'movie+subject'
        'batch_size': 32,
        'n_epochs': 50,  # Increased for S4-only training
        'learning_rate': 1e-3,
        'grad_clip_norm': 1.0,
        'log_interval': 5,
        'use_ema': True,
        'ema_decay': 0.98,  # Less aggressive EMA
        'patience': 15,     # More patience for S4 learning,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    logger.info(f"Using device: {config['device']}")
    
    # Load data
    logger.info(f"Loading HBN-EEG data from Release {config['release']}...")
    data_manager = SimpleHBNManager()
    subjects = data_manager.get_available_subjects(release=config['release'])
    logger.info(f"Found {len(subjects)} subjects")
    
    # Create dataset
    dataset = ICSSPMovieSegmentDataset(
        subjects=subjects,
        manager=data_manager,
        segment_length_sec=config['segment_length_sec'],
        overlap_ratio=config['overlap_ratio'],
        max_subjects=config['max_subjects'],
        target_channels=config['target_channels'],
        label_mode=config['label_mode'],
        release=config['release']
    )
    
    # Update config with actual number of classes
    config['n_classes'] = dataset.n_classes
    logger.info(f"Dynamic class detection: {config['n_classes']} classes")
    
    # Split data
    logger.info("Splitting dataset...")
    train_indices, temp_indices = train_test_split(
        range(len(dataset)), test_size=0.4, random_state=42,
        stratify=dataset.labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42,
        stratify=dataset.labels[temp_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['batch_size'], shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=config['batch_size'], shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_indices)}")
    logger.info(f"Val samples: {len(val_indices)}")
    logger.info(f"Test samples: {len(test_indices)}")
    
    # Define models with dynamic n_classes - OPTIMIZED S4
    models = {}
    
    # Try to add optimized S4 model with error handling
    try:
        print("🧪 Creating Optimized S4 model...")
        print("  Applying efficiency optimizations:")
        print("    - Reduced state size: 64 → 16")
        print("    - Single layer: 4 → 1") 
        print("    - Input downsampling: 2x")
        print("    - Smaller model dimension: 256 → 128")
        
        s4_model = S4Classifier(
            input_dim=config['target_channels'],
            d_model=128, d_state=16, n_layers=1,  # Optimized parameters
            n_classes=config['n_classes'],
            downsample_factor=2  # Reduce sequence length
        )
        print("✅ Optimized S4 model created successfully")
        
        # Test S4 model with dummy data
        print("🧪 Testing optimized S4 forward pass...")
        dummy_input = torch.randn(2, config['target_channels'], 100)
        print(f"Dummy input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            test_output = s4_model(dummy_input)
            print("✅ Forward pass completed")
            print(f"Output shape: {test_output.shape}")
            print(f"Expected speedup: ~8x (2x downsampling + 4x state reduction)")
            
            if not torch.isnan(test_output).any():
                models['S4'] = s4_model
                logger.info("Optimized S4 model added successfully")
                print("✅ Optimized S4 model added to training pipeline")
            else:
                logger.warning("S4 model produces NaN outputs, skipping...")
                print("❌ S4 model produces NaNs, skipping")
                
    except Exception as e:
        logger.warning(f"S4 model initialization failed: {e}, skipping...")
        print(f"❌ S4 model failed: {e}")
        import traceback
        logger.warning(f"S4 traceback: {traceback.format_exc()}")
        print(f"Full traceback:\n{traceback.format_exc()}")
    
    # Train and evaluate all models
    trainer = ModelTrainer(
        device=config['device'],
        use_ema=config['use_ema'],
        grad_clip_norm=config['grad_clip_norm'],
        log_interval=config['log_interval']
    )
    
    # Pass EMA decay to trainer
    trainer.ema_decay = config.get('ema_decay', 0.995)
    
    all_results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Train model
            training_results = trainer.train_model(
                model, train_loader, val_loader, model_name,
                n_epochs=config['n_epochs'],
                lr=config['learning_rate'],
                patience=config.get('patience', 10)  # Use config patience
            )
            
            # Comprehensive evaluation
            eval_results = trainer.comprehensive_evaluation(
                training_results['model'], test_loader, model_name
            )
            
            # Combine results
            all_results[model_name] = {
                **training_results,
                **eval_results
            }
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Test Accuracy: {eval_results['accuracy']:.2f}%")
            logger.info(f"  F1-Score: {eval_results['f1_score']:.3f}")
            logger.info(f"  Parameters: {eval_results['parameters']:,}")
            logger.info(f"  Training Time: {training_results['training_time']:.1f}s")
            logger.info(f"  Inference Time: {eval_results['avg_inference_time_ms']:.1f}ms")
            logger.info(f"  Peak Memory: {eval_results['peak_memory_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to train {model_name}: {e}")
            logger.error(f"Skipping {model_name} and continuing...")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            continue
    
    # Generate visualizations
    logger.info("\n" + "="*50)
    logger.info("Generating Visualizations")
    logger.info("="*50)
    
    # Visualize S4 kernels if S4 model was trained
    for model_name, results in all_results.items():
        if model_name == 'S4' and 'model' in results:
            visualize_s4_kernels(results['model'], model_name, 'plots')
    
    # Create model comparison plots
    visualize_all_models_comparison(all_results, 'plots')
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export to CSV
    csv_file = export_results_to_csv(all_results, config, timestamp)
    
    # Save JSON results
    results_file = f"results/icassp_results_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    results_for_json = {}
    for model_name, results in all_results.items():
        results_for_json[model_name] = {
            k: v for k, v in results.items() 
            if k not in ['model', 'train_losses', 'val_accuracies', 'learning_rates']
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results_for_json,
            'timestamp': timestamp
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    print(f"{'Model':<12} {'Accuracy':<10} {'F1-Score':<10} {'Parameters':<12} {'Train Time':<12} {'Inference':<10} {'Memory':<10}")
    print("-" * 90)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<12} "
              f"{results['accuracy']:<10.1f} "
              f"{results['f1_score']:<10.3f} "
              f"{results['parameters']:<12,} "
              f"{results['training_time']:<12.1f} "
              f"{results['avg_inference_time_ms']:<10.1f} "
              f"{results['peak_memory_mb']:<10.1f}")
    
    logger.info(f"\nCSV exported to: {csv_file}")
    logger.info("ICASSP Benchmarking Complete!")
    
    return all_results

def visualize_s4_kernels(model, model_name, save_dir='plots', sampling_rate=250):
    """Visualize S4 convolution kernels and frequency responses"""
    
    if not hasattr(model, 's4_layers'):
        logger.warning(f"Model {model_name} doesn't have S4 layers for visualization")
        return
    
    logger.info(f"Generating S4 kernel visualizations for {model_name}...")
    
    # Create frequency array for frequency response
    freqs = np.linspace(0, sampling_rate/2, 1024)  # Up to Nyquist frequency
    
    # Set up the plot
    n_layers = len(model.s4_layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    for layer_idx, s4_layer in enumerate(model.s4_layers):
        # Get impulse response (kernel)
        kernel = s4_layer.get_kernel(length=500)  # Shape: (500, d_model)
        
        # Get frequency response
        freq_response = s4_layer.get_frequency_response(freqs, sampling_rate)  # Shape: (d_model, 1024)
        
        # Plot impulse response (time domain)
        ax_time = axes[0, layer_idx]
        time_axis = np.arange(kernel.shape[0]) / sampling_rate * 1000  # Convert to ms
        
        # Plot first few dimensions for clarity
        n_dims_to_plot = min(5, kernel.shape[1])
        for dim in range(n_dims_to_plot):
            kernel_cpu = kernel[:, dim].cpu().numpy()
            ax_time.plot(time_axis, kernel_cpu, alpha=0.7, label=f'Dim {dim+1}')
        
        ax_time.set_xlabel('Time (ms)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'S4 Layer {layer_idx+1} - Impulse Response')
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        
        # Plot frequency response
        ax_freq = axes[1, layer_idx]
        
        # Plot average frequency response across dimensions
        mean_freq_response = np.mean(freq_response, axis=0)
        std_freq_response = np.std(freq_response, axis=0)
        
        ax_freq.loglog(freqs, mean_freq_response, 'b-', linewidth=2, label='Mean Response')
        ax_freq.fill_between(freqs, 
                            mean_freq_response - std_freq_response,
                            mean_freq_response + std_freq_response,
                            alpha=0.3, color='blue', label='±1 Std')
        
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Magnitude')
        ax_freq.set_title(f'S4 Layer {layer_idx+1} - Frequency Response')
        ax_freq.grid(True, alpha=0.3, which='both')
        ax_freq.legend()
        ax_freq.set_xlim([1, sampling_rate/2])
    
    plt.tight_layout()
    save_path = f'{save_dir}/{model_name.lower()}_s4_kernels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"S4 kernel visualization saved to {save_path}")
    
    # Generate additional detailed analysis plot
    _plot_s4_detailed_analysis(model, model_name, save_dir, sampling_rate)

def _plot_s4_detailed_analysis(model, model_name, save_dir, sampling_rate):
    """Generate detailed S4 analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collect all S4 parameters
    all_A = []
    all_dt = []
    all_kernels = []
    
    for layer_idx, s4_layer in enumerate(model.s4_layers):
        all_A.append(s4_layer.A.detach().cpu().numpy())
        all_dt.append(s4_layer.dt.detach().cpu().numpy())
        
        kernel = s4_layer.get_kernel(length=200)
        all_kernels.append(kernel.cpu().numpy())
    
    # Plot 1: A matrix eigenvalues (stability analysis)
    ax1 = axes[0, 0]
    for layer_idx, A in enumerate(all_A):
        # For diagonal A, eigenvalues are just the diagonal elements
        eigenvals = A
        ax1.scatter(np.real(eigenvals), np.imag(eigenvals), 
                   alpha=0.7, label=f'Layer {layer_idx+1}', s=20)
    
    # Add unit circle for stability reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('S4 Eigenvalues (Stability Analysis)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: Time constants
    ax2 = axes[0, 1]
    for layer_idx, A in enumerate(all_A):
        time_constants = 1 / (np.abs(A) + 1e-8)  # Avoid division by zero
        ax2.hist(time_constants, bins=20, alpha=0.7, label=f'Layer {layer_idx+1}')
    
    ax2.set_xlabel('Time Constant')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Time Constants')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Kernel comparison across layers
    ax3 = axes[1, 0]
    time_axis = np.arange(all_kernels[0].shape[0]) / sampling_rate * 1000
    
    for layer_idx, kernel in enumerate(all_kernels):
        # Use first dimension for comparison
        kernel_1d = kernel[:, 0]
        ax3.plot(time_axis, kernel_1d, label=f'Layer {layer_idx+1}', linewidth=2)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('S4 Kernels Comparison (First Dimension)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Frequency content analysis
    ax4 = axes[1, 1]
    freqs = np.fft.fftfreq(all_kernels[0].shape[0], 1/sampling_rate)[:all_kernels[0].shape[0]//2]
    
    for layer_idx, kernel in enumerate(all_kernels):
        # Average across all dimensions
        avg_kernel = np.mean(kernel, axis=1)
        fft_kernel = np.fft.fft(avg_kernel)
        magnitude = np.abs(fft_kernel[:len(freqs)])
        
        ax4.loglog(freqs[1:], magnitude[1:], label=f'Layer {layer_idx+1}', linewidth=2)
    
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('S4 Kernel Frequency Content')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()
    ax4.set_xlim([1, sampling_rate/2])
    
    plt.tight_layout()
    save_path = f'{save_dir}/{model_name.lower()}_s4_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"S4 detailed analysis saved to {save_path}")

def visualize_all_models_comparison(all_results, save_dir='plots'):
    """Create comparison visualizations across all models"""
    
    logger.info("Creating model comparison visualizations...")
    
    # Create performance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(all_results.keys())
    accuracies = [all_results[model]['accuracy'] for model in models]
    f1_scores = [all_results[model]['f1_score'] for model in models]
    train_times = [all_results[model]['training_time'] for model in models]
    parameters = [all_results[model]['parameters'] for model in models]
    
    # Performance comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(models))
    bars = ax1.bar(x_pos, accuracies, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # F1-Score comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, f1_scores, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Model F1-Score Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, train_times, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Time Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, time in zip(bars, train_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
                f'{time:.0f}s', ha='center', va='bottom')
    
    # Parameter count comparison (log scale)
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, parameters, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)])
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Parameters')
    ax4.set_title('Model Size Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    for bar, params in zip(bars, parameters):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{params:,}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    save_path = f'{save_dir}/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model comparison visualization saved to {save_path}")

# ============================================================================
# VISUALIZATION AND ANALYSIS FUNCTIONS
# ============================================================================

def visualize_s4_kernels_by_segment_length(results_dict, save_path='plots/s4_kernels_by_segment.png', 
                                          segment_lengths=['8s', '16s', '32s', '64s']):
    """Visualize S4 kernels across different segment lengths for interpretability analysis
    
    Args:
        results_dict: Dictionary containing results for each segment length
        save_path: Path to save the plot
        segment_lengths: List of segment length strings to analyze (e.g., ['8s', '16s', '32s', '64s', '128s'])
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_segments = len(segment_lengths)
    n_cols = min(3, n_segments)  # Max 3 columns for better layout
    n_rows = (n_segments + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_segments == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    if n_segments > 1:
        axes = axes.flatten()

    segment_lengths = segment_lengths
    
    for i, seg_length in enumerate(segment_lengths):
        if seg_length in results_dict and 'S4' in results_dict[seg_length]:
            s4_results = results_dict[seg_length]['S4']
            
            # Check if we have a trained model with kernel extraction capability
            if hasattr(s4_results.get('model'), 's4_layers') and len(s4_results['model'].s4_layers) > 0:
                try:
                    # Get kernel from first S4 layer
                    s4_layer = s4_results['model'].s4_layers[0]
                    kernel = s4_layer.get_kernel(length=1000)  # 1000 samples = 4s at 250Hz
                    
                    # Plot first few dimensions
                    time_axis = np.arange(1000) / 250.0  # Convert to seconds
                    
                    for dim in range(min(3, kernel.shape[1])):  # Plot first 3 dimensions
                        kernel_data = kernel[:, dim].detach().cpu().numpy()
                        axes[i].plot(time_axis, kernel_data, alpha=0.7, label=f'Dim {dim+1}')
                    
                    axes[i].set_title(f'S4 Kernels - {seg_length} segments')
                    axes[i].set_xlabel('Time (seconds)')
                    axes[i].set_ylabel('Kernel Response')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                    
                    # Add performance annotation
                    if 'accuracy' in s4_results:
                        acc = s4_results['accuracy']
                        axes[i].text(0.02, 0.98, f'Acc: {acc:.1f}%', 
                                   transform=axes[i].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Kernel extraction failed:\n{str(e)}', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'S4 Kernels - {seg_length} segments (Failed)')
            else:
                axes[i].text(0.5, 0.5, 'No S4 model available', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'S4 Kernels - {seg_length} segments (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"S4 kernel visualization saved to {save_path}")

def create_performance_vs_segment_length_plot(results_dict, save_path='plots/performance_vs_segment_length.png',
                                             segment_lengths=['8s', '16s', '32s', '64s'],
                                             models=['CNN', 'LSTM', 'S4', 'S5', 'Transformer']):
    """Create the main plot: Accuracy vs Segment Length for all models
    
    Args:
        results_dict: Dictionary containing results for each segment length
        save_path: Path to save the plot
        segment_lengths: List of segment length strings (e.g., ['8s', '16s', '32s', '64s', '128s'])
        models: List of model names to include in the comparison
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert segment lengths to numeric for plotting
    seg_numeric = [int(seg.replace('s', '')) for seg in segment_lengths]
    
    plt.figure(figsize=(12, 8))
    
    for model in models:
        accuracies = []
        valid_segments = []
        
        for i, seg_length in enumerate(segment_lengths):
            if (seg_length in results_dict and 
                model in results_dict[seg_length] and 
                'accuracy' in results_dict[seg_length][model]):
                
                acc = results_dict[seg_length][model]['accuracy']
                accuracies.append(acc)
                valid_segments.append(seg_numeric[i])
        
        if accuracies:  # Only plot if we have data
            plt.plot(valid_segments, accuracies, 'o-', linewidth=2, markersize=8, label=model)
    
    plt.xlabel('Segment Length (seconds)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance vs Temporal Context Length\n(EEG Movie Classification)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log', base=2)  # Log scale for segment lengths
    plt.xticks(seg_numeric, segment_lengths)
    
    # Add annotations about the key finding
    plt.text(0.02, 0.98, 'Key Question: Do sequence models\ngain advantage at longer contexts?', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance vs segment length plot saved to {save_path}")

def create_efficiency_pareto_plot(results_dict, save_path='plots/efficiency_pareto.png',
                                 segment_lengths=['8s', '16s', '32s', '64s'],
                                 models=['CNN', 'LSTM', 'S4', 'S5', 'Transformer']):
    """Create Pareto frontier plot: Parameters vs Accuracy
    
    Args:
        results_dict: Dictionary containing results for each segment length
        save_path: Path to save the plot
        segment_lengths: List of segment length strings to analyze
        models: List of model names to include in the comparison
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 8))
    
    # Generate colors dynamically based on number of models
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    # Collect data across all segment lengths
    for i, model in enumerate(models):
        params = []
        accuracies = []
        segment_labels = []
        
        for seg_length in segment_lengths:
            if (seg_length in results_dict and 
                model in results_dict[seg_length] and 
                'accuracy' in results_dict[seg_length][model]):
                
                result = results_dict[seg_length][model]
                if 'model' in result and hasattr(result['model'], 'parameters'):
                    # Count parameters
                    param_count = sum(p.numel() for p in result['model'].parameters() if p.requires_grad)
                    params.append(param_count / 1e6)  # Convert to millions
                    accuracies.append(result['accuracy'])
                    segment_labels.append(f"{model}-{seg_length}")
        
        if params:  # Only plot if we have data
            plt.scatter(params, accuracies, s=100, alpha=0.7, color=colors[i], label=model)
            
            # Annotate points with segment length
            for j, (p, a) in enumerate(zip(params, accuracies)):
                seg = segment_lengths[j] if j < len(segment_lengths) else '?'
                plt.annotate(seg, (p, a), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
    
    plt.xlabel('Parameters (Millions)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Efficiency: Accuracy vs Parameters\n(Pareto Frontier Analysis)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add efficiency annotation
    plt.text(0.02, 0.98, 'Efficient models appear\nin upper-left region', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Efficiency Pareto plot saved to {save_path}")

def generate_icassp_analysis_plots(results_file, segment_lengths=['8s', '16s', '32s', '64s'],
                                  models=['CNN', 'LSTM', 'S4', 'S5', 'Transformer']):
    """Generate all three key plots for ICASSP paper
    
    Args:
        results_file: Path to JSON results file
        segment_lengths: List of segment length strings to analyze (e.g., ['8s', '16s', '32s', '64s', '128s'])
        models: List of model names to include in the comparison
    """
    
    import json
    import os
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    logger.info("🎨 Generating ICASSP analysis plots...")
    logger.info(f"   Segment lengths: {segment_lengths}")
    logger.info(f"   Models: {models}")
    
    # Plot 1: Main finding - Performance vs Segment Length
    create_performance_vs_segment_length_plot(results, segment_lengths=segment_lengths, models=models)
    
    # Plot 2: Efficiency analysis - Pareto frontier
    create_efficiency_pareto_plot(results, segment_lengths=segment_lengths, models=models)
    
    # Plot 3: Interpretability - S4 kernel evolution
    visualize_s4_kernels_by_segment_length(results, segment_lengths=segment_lengths)
    
    logger.info("✅ All ICASSP plots generated successfully!")
    logger.info("📊 Key plots for your paper:")
    logger.info("   1. plots/performance_vs_segment_length.png - Main finding")
    logger.info("   2. plots/efficiency_pareto.png - Model efficiency comparison") 
    logger.info("   3. plots/s4_kernels_by_segment.png - S4 interpretability")
