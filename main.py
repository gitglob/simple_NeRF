import os
from math import ceil
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import tensor2image, encode, sample_rays, volume_rendering, seed_everything, save, load
import matplotlib.pyplot as plt


def train(dataset, model, optimizer, config, save_path):
    """Train the NeRF."""
    model.train()

    # Define the loss function
    criterion = nn.MSELoss()
    
    # Iterate over the epochs
    for i in range(config.num_iter):
        epoch = ceil(i//len(dataset))
        # Parse 1 image
        # [W*H, C], [W*H, C], [W*H, C], [W*H, 2]
        (img_target_rgb, img_rays_o, img_rays_d, img_bounds) = dataset[i]

        # Render the rays of that image
        # [W*H, S, C], [W*H, S], [W*H, S, C]
        img_pts, img_z_vals, img_view_dirs = sample_rays(img_rays_o, img_rays_d, img_bounds, N_samples=8)
        WH, S, C = img_pts.shape # Number of rays / samples / channels

        # Merge sampling with image dimensions to create individual rays
        pts = img_pts.reshape(WH*S, C)                 # [W*H*S, C]
        view_dirs = img_view_dirs.reshape(WH*S, C)     # [W*H*S, C]

        # Apply positional encoding
        pts = encode(pts, model.num_freqs_pos)              # [W*H*S, Dp] (Dp = 3 + 2*3*Fp)
        view_dirs = encode(view_dirs, model.num_freqs_dir)  # [W*H*S, Dd] (Dd = 3 + 2*3*Fd)

        # Set optimizer gradient to 0
        optimizer.zero_grad()
        
        # Query the model with the sampled points
        rgb, sigma = model(pts, view_dirs)   # [W*H*S, 3], [W*H*S, 1]

        # Isolate the sampling dimension again, to perform volume rendering
        rgb = rgb.view(WH, S, 3)
        sigma = sigma.view(WH, S, 1)
        
        # Convert raw model outputs to RGB, depth, and accumulated opacity maps
        # [W*H, 3], [W*H], [W*H]
        rgb_map, depth_map, acc_map = volume_rendering(rgb, sigma.squeeze(-1), img_z_vals)

        # The loss is the total squared error between the rendered and true pixel colors for both the coarse and fine renderings
        loss = criterion(rgb_map, img_target_rgb)
                
        # Backpropagate to compute the gradients
        loss.backward()
        
        # Update the model parameters using the optimizer
        optimizer.step()
    
        # Log an image every `log_interval` iterations
        if i % config.log_interval == 0:
            # Log the loss
            wandb.log({"Loss": loss.item()})

            # Log target and rendered image
            with torch.no_grad():
                rendered_image = tensor2image(rgb_map, config.image_w, config.image_h)
                target_image = tensor2image(img_target_rgb, config.image_w, config.image_h)
            wandb.log({
                "Rendered Image": wandb.Image(rendered_image, caption=f"Rendered Image - Epoch {epoch}, Iter {i}"),
                "Target Image": wandb.Image(target_image, caption=f"Target Image - Epoch {epoch}, Iter {i}")
            })

            # Print the loss and save the model checkpoint
            print(f"Iter {i}, Epoch {epoch}, Loss: {loss.item()}")
            save(model, optimizer, save_path)

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return
    device = torch.device("cuda")
    
    # Dataset path
    scene = "object"
    dataset_path = os.path.join(os.path.curdir, "data", "custom", scene)

    # Initialize wandb
    project_name = f"simple_nerf-{scene}"

    # Initialize wandb
    run_id = "v0.0.0"
    wandb.init(project=project_name, 
            entity="gitglob", 
            resume='allow', 
            id=run_id)
    
    # Sweep parameters
    wandb.config.update({
        "num_iter": 100000,      # Number of epochs
        "lr": 5e-4,              # Learning rate
        "N_c": 64,               # Coarse sampling
        "N_f": 128,              # Fine sampling
        "image_w": 128,          # Image weight dim
        "image_h": 128,          # Image height dim
        "num_samples": 8,        # Number of samples across each ray
        "num_freqs_pos": 3,      # Number of position encoding frequencies
        "num_freqs_dir": 2,      # Number of direction encoding frequencies
        "log_interval": 100      # Logging interval
    }, allow_val_change=True)
    config = wandb.config
    
    # Seed for reproducability
    seed_everything()

    # Initialize the Dataset
    transform = transforms.Compose([
        transforms.Resize((config.image_w, config.image_h)),
        transforms.ToTensor(),
    ])
    img_dir = os.path.join(dataset_path, "images")
    poses_path = os.path.join(dataset_path, "poses_bounds.npy")
    dataset = SceneDataset(images_dir=img_dir, poses_bounds_file=poses_path, transform=transform)

    # Initialize model and optimizer
    model = NeRF(config.num_freqs_pos, config.num_freqs_dir, device=device).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/model_" + scene + ".pth"
    if os.path.exists(save_path):
        model, optimizer = load(model, optimizer, save_path)
        print(f"Model and optimizer loaded from {save_path}\n")
    else:
        print("No existing model...\n")

    # Train the model
    train(dataset, model, optimizer, config, save_path)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()