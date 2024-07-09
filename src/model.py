import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

   
class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) model for rendering 3D scenes.
    """
    def __init__(self, num_freqs_pos=6, num_freqs_dir=4, device=None):
        """
        Initialize the NeRF model.

        Args:
            num_freqs_pos (int): Number of frequency bands for positional encoding of spatial coordinates.
            num_freqs_dir (int): Number of frequency bands for positional encoding of viewing directions.
        """
        super(NeRF, self).__init__()
        self.device = device
        
        self.num_freqs_pos = num_freqs_pos
        self.num_freqs_dir = num_freqs_dir
                
        # Add 2 dimensions for sine, cosine and 3 for x,y,z
        self.input_dim_pos = 3 + num_freqs_pos * 2 * 3
        self.input_dim_dir = 3 + num_freqs_dir * 2 * 3
        
        # MLP to process input points
        # Coarse network layers
        self.coarse_layers = nn.ModuleList([
            nn.Linear(self.input_dim_pos, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256 + self.input_dim_pos, 256),  # Skip connection
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])
        
        # Output layers for sigma and intermediate feature
        self.sigma_output = nn.Linear(256, 1)
        self.feature_output = nn.Linear(256, 256)
        
        # Fine network layers
        self.fine_network = nn.Sequential(
            nn.Linear(256 + self.input_dim_dir, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Output RGB color
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the NeRF model using Xavier initialization.
        """
        for layer in self.coarse_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.sigma_output.weight)
        nn.init.zeros_(self.sigma_output.bias)
        
        nn.init.xavier_uniform_(self.feature_output.weight)
        nn.init.zeros_(self.feature_output.bias)
        
        for layer in self.fine_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, d):
        """
        Forward pass through the NeRF model.

        Args:
            x (torch.Tensor): Input spatial coordinates of shape (batch_size, 3).
            d (torch.Tensor): Input viewing directions of shape (batch_size, 3).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4) containing RGB color and volume density.
        """
        # Coarse network
        h = x
        for i, layer in enumerate(self.coarse_layers):
            if i == 5:  # Skip connection
                h = torch.cat([h, x], dim=-1)

            h = F.relu(layer(h))
        
        sigma = self.sigma_output(h)
        feature = self.feature_output(h)
        
        # Concatenate feature with directional encoding
        h = torch.cat([feature, d], dim=-1)
        
        # Fine network
        rgb = self.fine_network(h)
        
        return rgb, sigma
