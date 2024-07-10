import os
from math import ceil
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import normalize, tensor2image, save, load
from src.utils import encode, sample_rays, volume_rendering, seed_everything


def train(dataset, model, optimizer, config, save_path):
    """Train the NeRF."""
    model.train()

    # Define the loss function
    criterion = nn.MSELoss()
    
    # Iterate over the epochs
    B = config.batch_size
    for i in range(config.num_iter):
        # Exponentially decaying learning rate
        lr = config.lr_final + (config.lr - config.lr_final) * (1 - i / config.num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # An epoch is how many times we have parsed the entire dataset
        epoch = ceil(i//len(dataset))
        
        # Parse 1 image
        # [W*H, 3], [W*H, 3], [W*H, 3], [W*H, 2]
        (img_target_rgb, img_rays_o, img_rays_d, img_bounds) = dataset[i]

        # Render the rays of that image
        # [W*H, S, 3], [W*H, S], [W*H, S, 3]
        img_pts, img_z_vals, img_view_dirs = sample_rays(img_rays_o, img_rays_d, img_bounds, N_samples=config.num_samples)
        WH, S, C = img_pts.shape # Number of rays / samples / channels

        # Merge sampling with image dimensions to create individual rays
        z_vals = img_z_vals.flatten()                  # [W*H*S]
        pts = img_pts.reshape(WH*S, C)                 # [W*H*S, 3]
        view_dirs = img_view_dirs.reshape(WH*S, C)     # [W*H*S, 3]

        # Apply positional encoding
        pts = encode(pts, model.num_freqs_pos)              # [W*H*S, Dp] (Dp = 3 + 2*3*Fp)
        view_dirs = encode(view_dirs, model.num_freqs_dir)  # [W*H*S, Dd] (Dd = 3 + 2*3*Fd)

        # Normalize points and view directions
        pts = normalize(pts)                # [W*H*S, Dp] (Dp = 3 + 2*3*Fp) ~ [0 ... 1]
        view_dirs = normalize(view_dirs)    # [W*H*S, Dp] (Dp = 3 + 2*3*Fp) ~ [0 ... 1]
        
        # Initialize variables to accumulate results
        rgb_map_batches = []
        depth_map_batches = []
        acc_map_batches = []
        
        # Process rays in batches
        for b in range(0, WH * S, B):
            # Get the current batch
            pts_batch = pts[b : b + B]                 # [B, Dp]
            view_dirs_batch = view_dirs[b : b + B]     # [B, Dd]
            z_vals_batch = z_vals[b : b + B]           # [B]

            # Set optimizer gradient to 0
            optimizer.zero_grad()

            # Query the model with the sampled points
            # [B, 3], [B, 1]
            rgb_batch, sigma_batch = model(pts_batch, view_dirs_batch)

            # Isolate the sampling dimension again, to perform volume rendering
            rgb_batch = rgb_batch.view(-1, S, 3)        # [-1, S, 3]
            sigma_batch = sigma_batch.view(-1, S, 1)    # [-1, S, 1]
            z_vals_batch = z_vals_batch.view(-1, S)     # [-1, S]
            
            # Convert raw model outputs to RGB, depth, and accumulated opacity maps
            # [B // S, 3], [B // S], [B // S]
            rgb_map, depth_map, acc_map = volume_rendering(rgb_batch, sigma_batch.squeeze(-1), z_vals_batch)

            # Calculate loss for this batch
            target_rgb_batch = img_target_rgb[b//S : b//S + B//S]   # [B//S, 3]
            loss = criterion(rgb_map, target_rgb_batch)

            # Backpropagate to compute the gradients
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Update the model parameters using the optimizer
            optimizer.step()
            
            # Accumulate results
            rgb_map_batches.append(rgb_map)
            depth_map_batches.append(depth_map)
            acc_map_batches.append(acc_map)

        # Concatenate all batches to reconstruct the full image
        rgb_map = torch.cat(rgb_map_batches, dim=0).view(WH, 3)
        depth_map = torch.cat(depth_map_batches, dim=0).view(WH)
        acc_map = torch.cat(acc_map_batches, dim=0).view(WH)
    
        # Log an image every `log_interval` iterations
        if i % config.log_interval == 0:
            # Log the learning rate
            wandb.log({"Learning Rate": lr})
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
        "lr": 5e-4,              # Learning rate start
        "lr_final": 5e-5,        # Learning rate finish
        "batch_size": 4096,      # Batch size (number of rays per iteration)
        "N_c": 64,               # Coarse sampling
        "N_f": 128,              # Fine sampling
        "image_w": 128,          # Image weight dim
        "image_h": 128,          # Image height dim
        "num_samples": 64,       # Number of samples across each ray
        "num_freqs_pos": 10,     # Number of position encoding frequencies
        "num_freqs_dir": 4,      # Number of direction encoding frequencies
        "log_interval": 10       # Logging interval
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