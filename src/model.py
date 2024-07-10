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
        # Coarse networks
        self.coarse_network_1 = nn.Sequential(
            nn.Linear(self.input_dim_pos, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.coarse_network_2 = nn.Sequential(
            nn.Linear(256 + self.input_dim_pos, 256),  # Skip connection
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
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
        for net in [self.coarse_network_1, self.coarse_network_2, self.fine_network]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.sigma_output.weight)
        nn.init.zeros_(self.sigma_output.bias)
        
        nn.init.xavier_uniform_(self.feature_output.weight)
        nn.init.zeros_(self.feature_output.bias)
        
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
        h = self.coarse_network_1(x)
        skip_x = torch.cat([h, x], dim=-1)
        h = self.coarse_network_2(skip_x)
        
        # Extract volume density
        sigma = F.relu(self.sigma_output(h))

        # Extract the features that will be used for the color prediction
        feature = self.feature_output(h)
        
        # Concatenate feature with directional encoding
        h = torch.cat([feature, d], dim=-1)
        
        # Fine network
        rgb = self.fine_network(h)
        
        return rgb, sigma

    def print_status(self):
        # Collect all gradients
        total_gradients = []
        for param in self.parameters():
            if param.grad is not None:
                total_gradients.append(param.grad.view(-1))
        
        if total_gradients:
            all_gradients = torch.cat(total_gradients)
            mean_gradient = all_gradients.mean().item()
            std_gradient = all_gradients.std().item()
            min_gradient = all_gradients.min().item()
            max_gradient = all_gradients.max().item()
            print(f"Gradients - Mean: {mean_gradient}, Std: {std_gradient}, Min: {min_gradient}, Max: {max_gradient}")
        
        # Collect all weights
        total_weights = []
        for param in self.parameters():
            total_weights.append(param.view(-1))
        
        if total_weights:
            all_weights = torch.cat(total_weights)
            mean_weight = all_weights.mean().item()
            std_weight = all_weights.std().item()
            min_weight = all_weights.min().item()
            max_weight = all_weights.max().item()
            print(f"Weights - Mean: {mean_weight}, Std: {std_weight}, Min: {min_weight}, Max: {max_weight}")
