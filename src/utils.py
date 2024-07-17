import numpy as np
import random 
import matplotlib.pyplot as plt
from PIL import Image
import torch


def create_gif(image_list, output_filename, duration=200, loop=0):
    image_list[0].save(
        output_filename,
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop
    )

def normalize(x: torch.Tensor):
    """Normalizes the input tensor."""
    min_vals, _ = x.min(dim=1, keepdim=True)
    max_vals, _ = x.max(dim=1, keepdim=True)
    x_norm = (x - min_vals) / (max_vals - min_vals + 1e-10)

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

def save(model_c, model_f, optimizer, fpath):
    """Saves model checkpoint."""
    torch.save({
        "model_c": model_c.state_dict(),
        "model_f": model_f.state_dict(),
        "optimizer": optimizer.state_dict()
    }, fpath)

def load(model_c, model_f, optimizer, fpath):
    """Loads model checkpoint."""
    checkpoint = torch.load(fpath)
    model_c.load_state_dict(checkpoint["model_c"])
    model_f.load_state_dict(checkpoint["model_f"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model_c, model_f, optimizer

def tensor2PILimage(x, w, h):
    """Reshapes a tensor into an image ready to be logged by WandB."""
    # Reshape tensor to an image shape
    img = x.view(w, h, 3)

    # Ensure values are in range 0-1
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert tensors to numpy arrays and then to uint8
    img_int = (img_norm.detach().cpu().numpy() * 255).astype(np.uint8)

    # Convert np array to image
    img = Image.fromarray(img_int)

    return img

def tensor2image(x, w, h):
    """Reshapes a tensor into an image ready to be logged by WandB."""
    # Reshape tensor to an image shape
    img = x.view(w, h, 3)

    # Ensure values are in range 0-1
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)

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
    
def stratified_sampling(rays_o, rays_d, bounds, N_samples=64):
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
    z_vals = torch.linspace(0.0, 1.0, S, dtype=torch.float32).cuda()  # [S]
    z_vals = z_vals.unsqueeze(0).expand(R, S)   # [R, S]

    # Scale depth values to be within near and far
    z_vals = near * (1 - z_vals) + far * z_vals  # [R, S]

    # # Perturb sampling depths to introduce randomness (optional)
    # perturb = torch.rand_like(z_vals) * (far - near) / S
    # z_vals = z_vals + perturb  # [R, S]

    # Compute sample points along each ray
    rays_o_expanded = rays_o.unsqueeze(1).expand(R, S, 3)      # [R, S, 3]
    rays_d_expanded = rays_d.unsqueeze(1).expand(R, S, 3)      # [R, S, 3]
    z_vals_expanded = z_vals.unsqueeze(-1)                     # [R, S, 1]

    pts = rays_o_expanded + z_vals_expanded * rays_d_expanded  # [R, S, 3]

    # Compute view directions
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize ray directions
    view_dirs = view_dirs.unsqueeze(1).expand(R, S, 3)         # [R, S, 3]

    return pts, view_dirs, z_vals

def sample_rays_f(rays_o, rays_d, z_vals, Nf):
    """
    Renders rays by sampling points along each ray.
    
    Args:
        rays_o (torch.Tensor): Tensor [R, 3] containing ray origins.
        rays_d (torch.Tensor): Tensor [R, 3] containing ray directions.
        z_vals (torch.Tensor): Tensor [R] containing samples Z coordinates.
        Nf (int, optional): Number of sample points per ray.
        
    Returns:
        pts (torch.Tensor): Tensor [R, Nf, 3] containing the sampled points.
        view_dirs (torch.Tensor): Tensor [R, Nf, 3] containing the view directions.
    """
    R = rays_o.shape[0]

    # Compute sample points along each ray
    rays_o_expanded = rays_o.unsqueeze(1).expand(R, Nf, 3)    # [R, Nf, 3]
    rays_d_expanded = rays_d.unsqueeze(1).expand(R, Nf, 3)    # [R, Nf, 3]
    z_vals_expanded = z_vals.unsqueeze(-1)                    # [R, Nf, 1]
    pts = rays_o_expanded + z_vals_expanded * rays_d_expanded # [R, Nf, 3]

    # Compute view directions
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize ray directions
    view_dirs = view_dirs.unsqueeze(1).expand(R, Nf, 3)       # [R, Nf, 3]

    return pts, view_dirs

def sample_fine_z_vals(weights, z_vals, B, Nc, Nf):
    """Samples Z values for the fine model."""
    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)  # [B, Nc]

    # Compute the CDF from the weights
    cdf = torch.cumsum(weights, dim=-1)                        # [B, Nc]
    zeros = torch.zeros((B, 1)).cuda()                         # [B, 1]
    cdf = torch.cat([zeros, cdf], dim=-1)                      # [B, 1 + Nc]

    # Draw uniform samples
    u = torch.rand((B, Nf)).cuda()                             # [B, Nf]

    # Use the cdf to find the new z_vals using inverse transform sampling
    z_vals_fine = torch.zeros((B, Nf)).cuda()                  # [B, Nf]
    for i in range(B):
        # Find the indices in the CDF that correspond to the uniform samples
        inds = torch.searchsorted(cdf[i], u[i], right=True)    # [Nf]
        
        # Ensure the indices are within valid range
        below_cdf = torch.clamp(inds - 1, min=0, max=Nc)       # [Nf]
        above_cdf = torch.clamp(inds, min=0, max=Nc)           # [Nf]
        below_z = torch.clamp(inds - 1, min=0, max=Nc-1)       # [Nf]
        above_z = torch.clamp(inds, min=0, max=Nc-1)           # [Nf]

        # Stack the indices to gather the corresponding values
        inds_cdf = torch.stack([below_cdf, above_cdf], -1)     # [Nf, 2]
        inds_z = torch.stack([below_z, above_z], -1)           # [Nf, 2]

        # Gather the CDF values at the given indices
        _cdf = cdf[i].unsqueeze(0).expand(Nf, -1)              # [Nf, Nc+1]
        cdf_g = torch.gather(_cdf, 1, inds_cdf)                # [Nf, 2]
        
        # Gather the z values at the given indices
        _z_vals = z_vals[i].unsqueeze(0).expand(Nf, -1)        # [Nf, Nc]
        z_vals_g = torch.gather(_z_vals, 1, inds_z)            # [Nf, 2]

        # Calculate the denominator for interpolation, ensuring no division by zero
        denom = cdf_g[:, 1] - cdf_g[:, 0]                      # [Nf, 2]
        denom[denom<1e-10] = 1
        
        # Calculate the interpolation factor t
        t = (u[i] - cdf_g[:, 0]) / denom                       # [Nf, 2]

        # Perform the linear interpolation to find the new z values
        z_vals_fine[i] = z_vals_g[:, 0] + t * (z_vals_g[:, 1] - z_vals_g[:, 0])

    return z_vals_fine

def hierarchical_sampling(rgb, sigma, z_vals, Nf):
    """
    Perform volume rendering to produce RGB, depth, and opacity maps.
    
    Parameters:
        rgb (tensor): The output color of the NeRF model with shape (B, Nc, 3).
        sigma (tensor): The output density of the NeRF model with shape (B, Nc, 1).
        z_vals (tensor): The z values (sample points) with shape (B, Nc).
        Nf (int): The number of fine Z coordinates to generate
        
    Returns:
        rgb_map (tensor): The rendered RGB image with shape (B, 3).
    """
    B, Nc, _ = rgb.shape
        
    # Calculate distances between adjacent z_vals
    deltas = z_vals[:, 1:] - z_vals[:, :-1]                    # [B, Nc-1]
    last_delta = 1e10 * torch.ones((B, 1)).cuda()              # [B, 1]
    deltas = torch.cat([deltas, last_delta], dim=1)            # [B, Nc]

    # Calculate alpha values (opacity) from sigma
    alpha = 1.0 - torch.exp(-sigma * deltas)                   # [B, Nc]
    
    # Calculate weights
    ones = torch.ones((B, 1)).cuda()                           # [B, 1]
    _ = torch.cat([ones, 1.0 - alpha + 1e-10], dim=1)          # [B, 1 + Nc]
    T = torch.cumprod(_, dim=-1)[:, :-1]                       # [B, Nc]
    weights = alpha * T                                        # [B, Nc]

    # Calculate the Z coordinates for the fine model
    z_vals_f = sample_fine_z_vals(weights.detach(), z_vals, B, Nc, Nf) # [B, Nf]

    # Compute the RGB map by summing the weighted colors
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)            # [B, 3]

    return rgb_map, z_vals_f

def volume_rendering(rgb, sigma, z_vals):
    """
    Perform volume rendering to produce RGB, depth, and opacity maps.
    
    Parameters:
        rgb (tensor): The output color of the NeRF model with shape (B, S, 3).
        sigma (tensor): The output density of the NeRF model with shape (B, S, 1).
        z_vals (tensor): The z values (sample points) with shape (B, S).
        
    Returns:
        rgb_map (tensor): The rendered RGB image with shape (B, 3).
    """
    B, Nf, _ = rgb.shape

    # Compute deltas (differences between adjacent z values)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    last_delta = 1e10 * torch.ones((B, 1)).cuda()           # [B, 1]
    deltas = torch.cat((deltas, last_delta), dim=1)         # [B, Nc]

    # Compute alpha values (1 - exp(-sigma * deltas))
    alpha = 1.0 - torch.exp(-sigma * deltas)                # [B, Nc]

    # Compute T values (cumulative product of (1 - alpha) terms)
    ones = torch.ones(B, 1).cuda()                          # [B, 1]
    _ = torch.cat([ones, 1.0 - alpha + 1e-10], dim=1)       # [B, 1 + Nc]
    T = torch.cumprod(_, dim=1)[:, :-1]                     # [B, Nc]

    # Compute weights
    weights = alpha * T                                     # [B, Nc]

    # Compute the RGB map by summing the weighted colors
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1) # [B, 3]

    # # Compute the depth map by summing the weighted z values
    # depth_map = torch.sum(weights * z_vals, dim=1)

    # # Compute the opacity map by summing the weights
    # opacity_map = torch.sum(weights, dim=1)
    
    return rgb_map
