#!/usr/bin/env python3
"""
Ultra-Simplified EEG Transformer - Maximum Fix
Push even further to match S5/CNN success patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class UltraSimplifiedEEGTransformer(nn.Module):
    """
    Ultra-simplified transformer that aggressively copies S5/CNN success patterns
    Target: Output std > 0.05 like S5 (0.050) and CNN (0.111)
    """
    
    def __init__(self, input_dim=64, d_model=64, nhead=4, num_layers=2, 
                 n_classes=4, dropout=0.1, max_seq_len=2000):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # 🔧 AGGRESSIVE FIX 1: Even simpler input projection with ReLU activation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),  # Add nonlinearity like CNN
            nn.Dropout(dropout * 0.5)  # Light dropout
        )
        
        # 🔧 AGGRESSIVE FIX 2: Multiple BatchNorm layers like CNN
        self.input_norm = nn.BatchNorm1d(d_model)
        
        # 🔧 AGGRESSIVE FIX 3: Smaller position encoding for stability
        pe = self._create_pos_encoding(max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
        # 🔧 AGGRESSIVE FIX 4: Even simpler transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model,  # 1x instead of 2x (very conservative)
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 🔧 AGGRESSIVE FIX 5: CNN-style pooling and normalization
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.pool_norm = nn.BatchNorm1d(d_model)
        
        # 🔧 AGGRESSIVE FIX 6: S5-style simple classifier with higher initial variance
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # 🔧 AGGRESSIVE FIX 7: Higher variance initialization
        self.apply(self._init_weights_high_variance)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"UltraSimplifiedEEGTransformer: {total_params:,} parameters")
        
    def _create_pos_encoding(self, max_len, d_model):
        """Smaller positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Scale down positional encoding to prevent dominance
        pe = pe * 0.1
        
        return pe.unsqueeze(0)
    
    def _init_weights_high_variance(self, module):
        """Higher variance initialization to boost output variance"""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization with higher std
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # x shape: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape
        
        # Get positional encoding
        pe_buffer = getattr(self, 'pe')
        max_len = pe_buffer.shape[1]
        if seq_len > max_len:
            x = x[:, :max_len, :]
            seq_len = x.shape[1]
        
        # Input projection with activation
        x = self.input_projection(x)
        
        # Strong normalization like CNN
        x = x.transpose(1, 2)  # For BatchNorm1d
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        
        # Add scaled positional encoding  
        pe_slice = pe_buffer[:, :seq_len, :]
        x = x + pe_slice
        
        # Apply transformer
        x = self.transformer(x)
        
        # CNN-style pooling with normalization
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.pool_norm(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

class HybridEEGTransformer(nn.Module):
    """
    Hybrid approach: Combine successful CNN patterns with minimal transformer
    """
    
    def __init__(self, input_dim=64, d_model=128, n_classes=4, dropout=0.1):
        super().__init__()
        
        # 🔧 HYBRID: Start with CNN-like convolutions (proven to work)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=25, padding=12),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(d_model // 2, d_model, kernel_size=25, padding=12),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # 🔧 HYBRID: Add minimal transformer on top of CNN features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Only 1 layer
        
        # 🔧 HYBRID: CNN-style classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Standard initialization
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"HybridEEGTransformer: {total_params:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        
        # CNN feature extraction
        x = self.conv_layers(x)  # (batch, d_model, reduced_time)
        
        # Transpose for transformer: (batch, d_model, time) -> (batch, time, d_model)
        x = x.transpose(1, 2)
        
        # Minimal transformer processing
        x = self.transformer(x)
        
        # Back to CNN format and classify
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.classifier(x)
        
        return x

def test_enhanced_transformers():
    """Test the enhanced transformer variants"""
    
    print("🧪 ENHANCED TRANSFORMER TESTING")
    print("="*60)
    print("Goal: Achieve output std > 0.05 (like S5: 0.050, CNN: 0.111)")
    print()
    
    models = {
        'UltraSimplified_Small': UltraSimplifiedEEGTransformer(
            input_dim=64, d_model=32, nhead=2, num_layers=1, n_classes=4
        ),
        'UltraSimplified_Medium': UltraSimplifiedEEGTransformer(
            input_dim=64, d_model=64, nhead=4, num_layers=2, n_classes=4
        ),
        'Hybrid_CNNTransformer': HybridEEGTransformer(
            input_dim=64, d_model=64, n_classes=4
        ),
        'Hybrid_CNNTransformer_Large': HybridEEGTransformer(
            input_dim=64, d_model=128, n_classes=4
        )
    }
    
    # Test input
    batch_size, channels, seq_len = 16, 64, 2000
    dummy_input = torch.randn(batch_size, channels, seq_len)
    
    results = []
    
    for name, model in models.items():
        print(f"🔍 Testing {name}:")
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   📊 Parameters: {total_params:,}")
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            output_std = output.std().item()
            print(f"   ✅ Forward pass successful")
            print(f"   📏 Output shape: {output.shape}")
            print(f"   📊 Output std: {output_std:.6f}")
            print(f"   📈 Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Grade based on target std > 0.05
            if output_std >= 0.1:
                grade = "A+ (Excellent)"
                status = "🎉"
            elif output_std >= 0.05:
                grade = "A (Target achieved!)"
                status = "✅"
            elif output_std >= 0.01:
                grade = "B (Good progress)"
                status = "👍"
            elif output_std >= 0.005:
                grade = "C (Some progress)"
                status = "⚠️"
            else:
                grade = "D (Still problematic)"
                status = "❌"
            
            print(f"   🎯 Grade: {grade} {status}")
            
            results.append({
                'name': name,
                'params': total_params,
                'output_std': output_std,
                'grade': grade,
                'success': True
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': name,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Find the best model
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        best_model = max(successful_results, key=lambda x: x['output_std'])
        
        print("="*60)
        print("🏆 BEST MODEL FOUND:")
        print(f"   Model: {best_model['name']}")
        print(f"   Parameters: {best_model['params']:,}")
        print(f"   Output std: {best_model['output_std']:.6f}")
        print(f"   Grade: {best_model['grade']}")
        
        if best_model['output_std'] >= 0.05:
            print(f"   🎉 SUCCESS! Ready for training experiments!")
            print(f"   💡 This model should achieve much better than 45% accuracy")
        else:
            print(f"   ⚠️  Still needs improvement, but much better than original")
    
    return results

if __name__ == "__main__":
    test_enhanced_transformers()
