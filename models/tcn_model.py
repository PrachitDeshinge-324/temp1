import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) * dilation // 2, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size-1) * dilation // 2, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)

class GaitTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128, 256], kernel_size=3):
        super(GaitTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # First convolution to match input dimensions
        self.conv_first = nn.Conv1d(input_dim, num_channels[0], kernel_size=1)
        
        # TCN blocks with increasing dilation
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            
            layers.append(ResidualBlock(
                in_channels, out_channels, kernel_size, dilation=dilation_size
            ))
        
        self.tcn_network = nn.Sequential(*layers)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Convert to (batch_size, features, sequence_length) for 1D convolution
        x = x.transpose(1, 2)
        
        # Initial convolution to match dimensions
        x = self.conv_first(x)
        
        # Apply TCN blocks
        x = self.tcn_network(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.fc(x)
        
        return x