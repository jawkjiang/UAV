"""
Multi-Model Architecture for GPS Spoofing Detection

Implements 7 different deep learning architectures:
1. 1D-CNN (Baseline from step1)
2. LSTM
3. BiLSTM (Bidirectional LSTM)
4. GRU
5. CNN-LSTM Hybrid
6. TCN (Temporal Convolutional Network from step1)
7. Transformer

All models follow the same interface:
- Input: [batch_size, 50, 13]
- Output: [batch_size, 1] with values in [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# BASELINE: 1D-CNN (Copied from step1_benchmark)
# ============================================================================

class Conv1DBlock(nn.Module):
    """Basic 1D convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GPSSpoofingDetector(nn.Module):
    """
    1D-CNN model for GPS spoofing detection.
    
    REQUIRED INTERFACE - DO NOT CHANGE
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        """
        REQUIRED SIGNATURE - DO NOT CHANGE PARAMETER NAMES OR ORDER
        
        Args:
            n_features: Number of input features (always 13)
            window_size: Window size (always 50)
            dropout: Dropout probability (default 0.3)
        """
        super(GPSSpoofingDetector, self).__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        
        # Convolutional layers
        self.conv1 = Conv1DBlock(n_features, 64, kernel_size=7, padding=3)
        self.conv2 = Conv1DBlock(64, 128, kernel_size=5, padding=2)
        self.conv3 = Conv1DBlock(128, 256, kernel_size=3, padding=1)
        self.conv4 = Conv1DBlock(256, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for FC layers
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.bn_fc2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        REQUIRED SIGNATURE - DO NOT CHANGE
        
        Args:
            x: Input tensor of shape [batch_size, window_size, n_features]
               = [batch_size, 50, 13]
        
        Returns:
            Output tensor of shape [batch_size, 1]
            Values MUST be in range [0, 1] (use sigmoid)
        """
        # Transpose to [batch, features, time]
        x = x.transpose(1, 2)
        
        # Convolutional layers with pooling
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# LSTM
# ============================================================================

class LSTMDetector(nn.Module):
    """
    LSTM-based detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    LSTM(hidden_size=128, num_layers=2, dropout=dropout)
      ↓
    Take last hidden state: [batch, 128]
      ↓
    FC: 128 → 64 → 32 → 1
      ↓
    Sigmoid → [batch, 1]
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(LSTMDetector, self).__init__()
        
        self.hidden_size = 128
        self.num_layers = 2
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, 50, 128]
        # h_n: [num_layers, batch, 128]
        
        # Take last hidden state from last layer
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# BiLSTM
# ============================================================================

class BiLSTMDetector(nn.Module):
    """
    Bidirectional LSTM detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    BiLSTM(hidden_size=64, num_layers=2)
      ↓
    Concat forward & backward: [batch, 128]
      ↓
    FC: 128 → 64 → 32 → 1
      ↓
    Sigmoid → [batch, 1]
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(BiLSTMDetector, self).__init__()
        
        self.hidden_size = 64  # Per direction
        self.num_layers = 2
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0,
            bidirectional=True  # KEY: bidirectional
        )
        
        # FC layers (input is hidden_size * 2 due to bidirectional)
        self.fc1 = nn.Linear(self.hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # BiLSTM forward
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # h_n: [num_layers * 2, batch, 64]
        
        # Concatenate forward and backward from last layer
        # h_n[-2]: forward direction last layer
        # h_n[-1]: backward direction last layer
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# GRU
# ============================================================================

class GRUDetector(nn.Module):
    """
    GRU-based detector.
    
    Similar to LSTM but more efficient.
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(GRUDetector, self).__init__()
        
        self.hidden_size = 128
        self.num_layers = 2
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        # FC layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # GRU forward
        gru_out, h_n = self.gru(x)
        # h_n: [num_layers, batch, 128]
        
        # Take last hidden state
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# CNN-LSTM Hybrid
# ============================================================================

class CNNLSTMDetector(nn.Module):
    """
    Hybrid CNN-LSTM detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    1D Conv layers (extract local features)
      ↓  [batch, seq_len', 128]
    LSTM (model temporal dependencies)
      ↓  [batch, 128]
    FC → Sigmoid
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(CNNLSTMDetector, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # After 2 pooling layers: seq_len = 50 / 4 = 12
        # LSTM on extracted features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # FC layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn_fc = nn.BatchNorm1d(64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # Transpose for Conv1d: [batch, 13, 50]
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 64, 25]
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 128, 12]
        
        # Transpose back for LSTM: [batch, 12, 128]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# TCN (Copied from step1_benchmark)
# ============================================================================

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) architecture.
    
    Uses dilated convolutions for larger receptive field.
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        """
        Initialize TCN.
        
        Args:
            n_features: Number of input features
            window_size: Length of input window
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        
        # Fixed architecture
        num_channels = [64, 128, 128]
        kernel_size = 3
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, window_size, n_features]
        
        Returns:
            [batch_size, 1] probabilities
        """
        # Transpose to [batch, features, time]
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        x = self.network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# Transformer
# ============================================================================

class TransformerDetector(nn.Module):
    """
    Transformer-based detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    Linear projection to d_model
      ↓
    Positional encoding
      ↓
    Transformer encoder (4 heads, 3 layers)
      ↓
    Global average pooling
      ↓
    FC → Sigmoid
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(TransformerDetector, self).__init__()
        
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 3
        self.dim_feedforward = 128
        
        # Input projection
        self.input_projection = nn.Linear(n_features, self.d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(window_size, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(self.d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(32)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # Project to d_model
        x = self.input_projection(x)  # [batch, 50, 64]
        
        # Add positional encoding
        batch_size = x.size(0)
        pos_enc = self.pos_encoding.to(x.device)
        x = x + pos_enc  # Broadcasting
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, 50, 64]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch, 64]
        
        # Output layers
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x


# ============================================================================
# Factory Function
# ============================================================================

def create_model(model_type: str, 
                n_features: int, 
                window_size: int = 50, 
                dropout: float = 0.3) -> nn.Module:
    """
    Factory function to create model instances.
    
    REQUIRED SIGNATURE - DO NOT CHANGE
    
    Args:
        model_type: One of ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
        n_features: Number of input features (always 13)
        window_size: Window size (always 50)
        dropout: Dropout probability
    
    Returns:
        Model instance conforming to required interface
    """
    models = {
        'cnn': GPSSpoofingDetector,
        'lstm': LSTMDetector,
        'bilstm': BiLSTMDetector,
        'gru': GRUDetector,
        'cnn_lstm': CNNLSTMDetector,
        'tcn': TemporalConvNet,
        'transformer': TransformerDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Must be one of {list(models.keys())}")
    
    return models[model_type](n_features, window_size, dropout)
