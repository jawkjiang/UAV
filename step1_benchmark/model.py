"""
1D-CNN Model for GPS Spoofing Detection

Implements a 1D Convolutional Neural Network for temporal feature extraction
from UAV sensor windows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """
    Basic 1D convolution block with BatchNorm and ReLU.
    """
    
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
    
    Architecture:
    - Multiple 1D convolutional layers for temporal feature extraction
    - Global pooling to aggregate temporal information
    - Fully connected layers for classification
    - Single output: probability of GPS spoofing at window end
    """
    
    def __init__(self, n_features, window_size=50, dropout=0.3):
        """
        Initialize model.
        
        Args:
            n_features: Number of input features per time step
            window_size: Length of input window
            dropout: Dropout probability
        """
        super(GPSSpoofingDetector, self).__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        
        # Convolutional layers
        # Input shape: [batch, n_features, window_size]
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
        self.fc1 = nn.Linear(128 * 2, 64)  # *2 for avg and max pooling
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for FC layers
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.bn_fc2 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, window_size, n_features]
        
        Returns:
            Output tensor of shape [batch_size, 1] with spoofing probabilities
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
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch, 128]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch, 128]
        
        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)  # [batch, 256]
        
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
        
        # Sigmoid for binary classification
        x = torch.sigmoid(x)
        
        return x


class TemporalConvNet(nn.Module):
    """
    Alternative Temporal Convolutional Network (TCN) architecture.
    
    Uses dilated convolutions for larger receptive field.
    """
    
    def __init__(self, n_features, window_size=50, num_channels=[64, 128, 128], 
                 kernel_size=3, dropout=0.3):
        """
        Initialize TCN.
        
        Args:
            n_features: Number of input features
            window_size: Length of input window
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        
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
    
    def forward(self, x):
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


def create_model(model_type='cnn', n_features=13, window_size=50, dropout=0.3):
    """
    Factory function to create model.
    
    Args:
        model_type: 'cnn' or 'tcn'
        n_features: Number of input features
        window_size: Window size
        dropout: Dropout probability
    
    Returns:
        Model instance
    """
    if model_type == 'cnn':
        return GPSSpoofingDetector(n_features, window_size, dropout)
    elif model_type == 'tcn':
        return TemporalConvNet(n_features, window_size, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
