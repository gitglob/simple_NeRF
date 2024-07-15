import torch
import torch.nn as nn
import torch.nn.functional as F

   
class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) model for rendering 3D scenes.
    """
    def __init__(self, fp=6, fd=4):
        """
        Initialize the NeRF model.

        Args:
            fp (int): Number of frequency bands for positional encoding of spatial coordinates.
            fd (int): Number of frequency bands for positional encoding of viewing directions.
        """
        super(NeRF, self).__init__()
        
        self.fp = fp
        self.fd = fd
                
        # Add 2 dimensions for sine, cosine and 3 for x,y,z
        self.input_dim_pos = 3 + fp * 2 * 3
        self.input_dim_dir = 3 + fd * 2 * 3
        
        # MLP to process input points
        # Coarse networks
        self.fc1 = nn.Linear(self.input_dim_pos, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        
        self.fc6 = nn.Linear(256 + self.input_dim_pos, 256)  # Skip connection
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        
        # Output layers for sigma and intermediate feature
        self.sigma_head = nn.Linear(256, 1)
        self.feature_head = nn.Linear(256, 256)
        
        # Fine network layers
        self.fc11 = nn.Linear(256 + self.input_dim_dir, 128)
        self.fc12 = nn.Linear(128, 3)  # Output RGB color
        
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the NeRF model using Xavier initialization.
        """
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, 
                      self.fc5, self.fc6, self.fc7, self.fc8, 
                      self.fc9, self.fc10, 
                      self.sigma_head, self.feature_head,
                      self.fc11, self.fc12]:
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
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        
        skip_x = torch.cat([h, x], dim=-1)
        h = F.relu(self.fc6(skip_x))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))
        h = F.relu(self.fc9(h))
        h = F.relu(self.fc10(h))
        
        # Extract volume density
        sigma = F.relu(self.sigma_head(h))

        # Extract the features that will be used for the color prediction
        feature = self.feature_head(h)
        
        # Concatenate feature with directional encoding
        h = torch.cat([feature, d], dim=-1)
        
        # Fine network
        h = F.relu(self.fc11(h))
        rgb = F.sigmoid(self.fc12(h))
        
        return rgb, sigma
