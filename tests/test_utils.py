import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from src.utils import sample_rays


def test_sample_rays():
    # Create test inputs
    B = 2  # Batch size
    R = 4  # Number of rays
    C = 3  # Color channels (3 for RGB)
    
    # Define ray origins, directions, and bounds
    rays_o = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], 
                           [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]])  # [B, R, C]
    rays_d = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 
                           [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # [B, R, C]
    bounds = torch.tensor([[1.0, 3.0], [2.0, 4.0]])  # [B, 2]
    print(f"Origin Ray shape [{rays_o.shape}]:")
    print(rays_o)
    print(f"Direction Ray shape [{rays_d.shape}]:")
    print(rays_d)

    # Call the function
    N_samples = 4  # Use a small number for simplicity
    pts, z_vals, view_dirs = sample_rays(rays_o, rays_d, bounds, N_samples)

    # Print the outputs
    print(f"Sampled Points [{pts.shape}]:")
    print(pts)
    print(f"\nDepth Values [{z_vals.shape}]:")
    print(z_vals)
    print(f"\nView Directions [{view_dirs.shape}]:")
    print(view_dirs)

def run_tests():
    test_sample_rays()

if __name__ == "__main__":
    run_tests()