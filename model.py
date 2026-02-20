"""CNN model architecture for chess move prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN_FILTERS, CNN_KERNEL_SIZES, DENSE_LAYERS, DROPOUT_RATE, BOARD_CHANNELS


class ChessCNN(nn.Module):
    """Convolutional Neural Network for chess move prediction."""

    def __init__(self, num_output_moves=4672):
        """
        Initialize Chess CNN.
        
        Args:
            num_output_moves: Number of possible moves (8x8x8x8 + castling + promotions ≈ 4672)
        """
        super(ChessCNN, self).__init__()
        
        # Input: (batch, BOARD_CHANNELS, 8, 8)
        self.conv_layers = nn.ModuleList()
        in_channels = BOARD_CHANNELS
        
        # Build convolutional blocks
        for out_channels, kernel_size in zip(CNN_FILTERS, CNN_KERNEL_SIZES):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels
        
        # Adaptive pooling to fixed size for fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in = CNN_FILTERS[-1]
        
        for fc_out in DENSE_LAYERS:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(DROPOUT_RATE)
            ))
            fc_in = fc_out
        
        # Output layer
        self.output = nn.Linear(fc_in, num_output_moves)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, BOARD_CHANNELS, 8, 8)
            
        Returns:
            Logits of shape (batch, num_output_moves)
        """
        # Convolutional blocks
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output
        x = self.output(x)
        return x


class ChessCNNv2(nn.Module):
    """Improved CNN with residual-like connections."""

    def __init__(self, num_output_moves=4672):
        """
        Initialize improved Chess CNN.
        
        Args:
            num_output_moves: Number of possible moves
        """
        super(ChessCNNv2, self).__init__()
        
        # Initial conv block
        self.initial = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(64, 128),
            self._make_block(128, 256),
            self._make_block(256, 256),
        ])
        
        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_output_moves)
        )

    def _make_block(self, in_channels, out_channels):
        """Create a conv block with stride 2 for downsampling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass."""
        x = self.initial(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.head(x)
        return x


if __name__ == "__main__":
    # Test model
    model = ChessCNN()
    x = torch.randn(2, BOARD_CHANNELS, 8, 8)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
