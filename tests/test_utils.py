import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scipy.interpolate import interp1d
import torch
from src.utils import stratified_sampling


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
    pts, z_vals, view_dirs = stratified_sampling(rays_o, rays_d, bounds, N_samples)

    # Print the outputs
    print(f"Sampled Points [{pts.shape}]:")
    print(pts)
    print(f"\nDepth Values [{z_vals.shape}]:")
    print(z_vals)
    print(f"\nView Directions [{view_dirs.shape}]:")
    print(view_dirs)

def test_inverse_transform_sampling():
    torch.manual_seed(42)
    B = 2
    S = 3
    print(f"B, S: ({B},{S})")

    # Sample data
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.25, 0.15], [0.1, 0.2, 0.3, 0.1, 0.3]], dtype=torch.float32)
    print("Weights:\n", weights.numpy())
    z_vals = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    print("Z:\n", z_vals.numpy())
    
    # Compute the CDF from the weights
    cdf = torch.cumsum(weights, dim=-1)
    
    # Ensure the last value of each row in the CDF is exactly 1
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Print the CDF for understanding
    print("CDF:\n", cdf.numpy())

    # Draw uniform samples
    u = torch.tensor([[0, 0.12, 0.95],
                      [0.1, 0.32, 1]])
    
    # Print the uniform samples for understanding
    print("Uniform samples (u):\n", u.numpy())

    # Use the cdf to find the new z_vals
    z_vals_fine = torch.zeros((B, S), device=z_vals.device)
    for i in range(B):
        # Find the indices in the CDF that correspond to the uniform samples
        inds = torch.searchsorted(cdf[i], u[i], right=True)
        print("Indices:\n", inds.numpy())

        # Ensure the indices are within valid range
        below_cdf = torch.clamp(inds - 1, min=0, max=cdf.shape[-1]-1)
        above_cdf = torch.clamp(inds, max=cdf.shape[-1] - 1)
        below_z = torch.clamp(inds - 1, min=0, max=z_vals.shape[-1]-1)
        above_z = torch.clamp(inds, max=z_vals.shape[-1] - 1)
        
        # Stack the indices to gather the corresponding values
        inds_g_z = torch.stack([below_z, above_z], -1)
        inds_g_cdf = torch.stack([below_cdf, above_cdf], -1)

        # Gather the CDF values at the given indices
        cdf_g = torch.gather(cdf[i].unsqueeze(0).expand(inds_g_cdf.shape[0], -1), 1, inds_g_cdf)
        
        # Gather the z values at the given indices
        z_vals_g = torch.gather(z_vals[i].unsqueeze(0).expand(inds_g_z.shape[0], -1), 1, inds_g_z)

        # Calculate the denominator for interpolation, ensuring no division by zero
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        
        # Calculate the interpolation factor t
        t = (u[i] - cdf_g[..., 0]) / denom

        # Perform the linear interpolation to find the new z values
        z_vals_fine[i] = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])

    # Print the fine z values for understanding
    print("Fine z values (z_vals_fine):\n", z_vals_fine.numpy())

def run_tests():
    # test_sample_rays()
    test_inverse_transform_sampling()

if __name__ == "__main__":
    run_tests()