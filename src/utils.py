import numpy as np
import random 
import matplotlib.pyplot as plt
import torch


def normalize(x: torch.Tensor):
    """Normalizes the input tensor."""
    min_vals, _ = x.min(dim=1, keepdim=True)
    max_vals, _ = x.max(dim=1, keepdim=True)
    x_norm = (x - min_vals) / (max_vals - min_vals + 1e-8)

    return x_norm

def show(img):
    plt.imshow(img.reshape((128, 128, 3)).detach().cpu().numpy())
    plt.savefig("image.png")

def seed_everything(seed=42):
    """Seeds everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save(model, optimizer, fpath):
    """Saves model checkpoint."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, fpath)

def load(model, optimizer, fpath):
    """Loads model checkpoint."""
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer

def tensor2image(x, w, h):
    """Reshapes a tensor into an image ready to be logged by WandB."""
    # Reshape tensor to an image shape
    img = x.view(w, h, 3)

    # Ensure values are in range 0-1
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert tensors to numpy arrays and then to uint8
    img_int = (img_norm.detach().cpu().numpy() * 255).astype(np.uint8)

    return img_int

def encode(x, encoding_dim):
    """Encodes x to a higher dimension using sines and cosines."""
    encoding = [x]
    for i in range(encoding_dim):
        for fn in [torch.sin, torch.cos]:
            encoding.append(fn(2.0 ** i * x))
    return torch.cat(encoding, dim=-1).float()

def batch_encode(x, encoding_dim):
    """Encodes a batch of tensors to a higher dimension using sines and cosines."""
    batch_encoding = []
    for batch in x:
        batch_encoding.append(encode(batch, encoding_dim))
    return torch.stack(batch_encoding)
    
def sample_rays(rays_o, rays_d, bounds, N_samples=64):
    """
    Renders rays by sampling points along each ray.
    
    Args:
        rays_o (torch.Tensor): Tensor [N_rays, color_channels] containing ray origins.
        rays_d (torch.Tensor): Tensor [N_rays, color_channels] containing ray directions.
        bounds (torch.Tensor): Tensor [N_rays, 2] containing near and far bounds.
        N_samples (int, optional): Number of sample points per ray. Default is 64.
        
    Returns:
        pts (torch.Tensor): Tensor [N_rays, N_samples, color_channels] containing the sampled points.
        z_vals (torch.Tensor): Tensor [N_rays, N_samples] containing the depth values.
        view_dirs (torch.Tensor): Tensor [N_rays, N_samples, 3] containing the view directions.
    """
    R, C = rays_o.shape
    S = N_samples

    # Compute near and far bounds for each ray
    near = bounds[..., 0].unsqueeze(-1)  # [R, 1]
    far = bounds[..., 1].unsqueeze(-1)   # [R, 1]

    # Sample depth values
    z_vals = torch.linspace(0.0, 1.0, S, dtype=torch.float32, device=rays_o.device)  # [S]
    z_vals = z_vals.unsqueeze(0).expand(R, S)   # [R, S]

    # Scale depth values to be within near and far
    z_vals = near * (1 - z_vals) + far * z_vals  # [R, S]

    # # Perturb sampling depths to introduce randomness (optional)
    # perturb = torch.rand_like(z_vals) * (far - near) / S
    # z_vals = z_vals + perturb  # [R, S]

    # Compute sample points along each ray
    rays_o_expanded = rays_o.unsqueeze(1).expand(R, S, 3)   # [R, S, 3]
    rays_d_expanded = rays_d.unsqueeze(1).expand(R, S, 3)   # [R, S, 3]
    z_vals_expanded = z_vals.unsqueeze(-1)                     # [R, S, 1]

    pts = rays_o_expanded + z_vals_expanded * rays_d_expanded  # [R, S, 3]

    # Compute view directions
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize ray directions
    view_dirs = view_dirs.unsqueeze(1).expand(R, S, 3)      # [R, S, 3]

    return pts, view_dirs, z_vals

def volume_rendering(rgb, sigma, z_vals):
    """
    Perform volume rendering to produce RGB, depth, and opacity maps.
    
    Parameters:
        rgb_sigma (tensor): The output of the NeRF model with shape (batch_size, num_samples, 4).
                            The last dimension contains RGB values and the density (sigma).
        z_vals (tensor): The z values (sample points) with shape (batch_size, num_samples).
        
    Returns:
        rgb_map (tensor): The rendered RGB image with shape (batch_size, 3).
        depth_map (tensor): The rendered depth map with shape (batch_size).
        opacity_map (tensor): The rendered opacity map with shape (batch_size).
    """
    batch_size = rgb.shape[0]
        
    # Calculate distances between adjacent z_vals
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    ones = torch.Tensor([1e10]).expand(dists.shape[0], 1).cuda()
    dists = torch.cat([dists, ones], dim=-1)  # [batch_size, num_samples]

    # Calculate alpha values (opacity) from sigma
    alpha = 1.0 - torch.exp(-sigma * dists)

    # Calculate weights
    ones = torch.ones((batch_size, 1)).cuda()
    T = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * T

    # Calculate RGB map
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    
    # Calculate depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # Calculate opacity map
    opacity_map = torch.sum(weights, dim=-1)
    
    return rgb_map, depth_map, opacity_map
