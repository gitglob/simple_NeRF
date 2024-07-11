import os
from math import ceil
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import tensor2image, save, load
from src.utils import encode, batch_encode, volume_rendering, seed_everything


def validate(val_dataloader, model, criterion, config):
    with torch.no_grad():
        # [1, W*H, 3], [1, W*H, S, 3], [1, W*H, S, 3], [1, W*H, S]
        target_rgb, pts, view_dirs, z_vals = next(iter(val_dataloader))
        target_rgb = target_rgb.squeeze(0)   # [W*H, 3]
        pts = pts.squeeze(0)                 # [W*H, S, 3]
        view_dirs = view_dirs.squeeze(0)     # [W*H, S, 3]
        z_vals = z_vals.squeeze(0)           # [W*H, S]
        pts = batch_encode(pts, model.num_freqs_pos)             # [W*H, S, Dp]
        view_dirs = batch_encode(view_dirs, model.num_freqs_dir) # [W*H, S, Dd]

        # Split validation forward pass to batches so that it can fit to CUDA memory
        B = config.batch_size
        N, S = z_vals.shape
        rendered_images = []
        target_images = []
        for start in range(0, N, B):
            # Get the batch tensors
            end = start + B
            pts_batch = pts[start:end]               # [B, S, Dp]
            view_dirs_batch = view_dirs[start:end]   # [B, S, Dd]
            z_vals_batch = z_vals[start:end]         # [B, S]
            target_rgb_batch = target_rgb[start:end] # [B, 3]

            # Merge Batch and Sample dimensions
            pts_batch = pts_batch.view(B*S, -1)              # [B*S, Dp]
            view_dirs_batch = view_dirs_batch.view(B*S, -1)  # [B*S, Dd]

            # [B*S, 3], [B*S, 1]
            rgb_batch, sigma_batch = model(pts_batch, view_dirs_batch)
            
            # Isolate the sampling dimension again, to perform volume rendering
            rgb_batch = rgb_batch.view(B, S, 3)               # [B, S, 3]
            sigma_batch = sigma_batch.squeeze(-1).view(B, S)  # [B, S]

            # [B, 3], [B, 3], [B, 3]
            rgb_map_batch, depth_map_batch, acc_map_batch = volume_rendering(rgb_batch, sigma_batch, z_vals_batch)
            
            # Concatenate validation batches
            rendered_images.append(rgb_map_batch)
            target_images.append(target_rgb_batch)

        rendered_image = torch.cat(rendered_images, dim=0)  # [W*H, 3]
        target_image = torch.cat(target_images, dim=0)      # [W*H, 3]
        
        # Calculate the validation loss
        loss = criterion(rendered_image, target_image)

        # Convert tensors to images
        rendered_image_np = tensor2image(rendered_image, config.image_w, config.image_h)
        target_image_np = tensor2image(target_image, config.image_w, config.image_h)

        return loss, rendered_image_np, target_image_np

def train(dataset, train_dataloader, val_dataloader, 
          model, optimizer, config, save_path):
    """Train the NeRF."""
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Iterate over the epochs
    B = config.batch_size
    for i in range(config.num_iter):
        dataset.set_mode("train")
        model.train()

        # Exponentially decaying learning rate
        lr = config.lr_final + (config.lr - config.lr_final) * (1 - i / config.num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Parse B random rays sampled at S points
        # [B, 3], [B, S, 3], [B, S, 3], [B, S]
        target_rgb, pts, view_dirs, z_vals = next(iter(train_dataloader))
        B, S, C = pts.shape

        # Apply positional encoding
        pts = batch_encode(pts, model.num_freqs_pos)              # [B, S, Dp] (Dp = 3 + 2*3*Fp)
        view_dirs = batch_encode(view_dirs, model.num_freqs_dir)  # [B, S, Dd] (Dd = 3 + 2*3*Fd)

        # Merge Batch and Sample dimensions
        pts = pts.view(B*S, -1)              # [B*S, Dp]
        view_dirs = view_dirs.view(B*S, -1)  # [B*S, Dd]

        # Normalize points and view directions
        # pts = normalize(pts)               # [B*S, Dp] (Dp = 3 + 2*3*Fp) ~ [0 ... 1]
        # view_dirs = normalize(view_dirs)   # [B*S, Dp] (Dp = 3 + 2*3*Fp) ~ [0 ... 1]

        # Set optimizer gradient to 0
        optimizer.zero_grad()

        # Query the model with the sampled points
        # [B*S, 3], [B*S, 1]
        rgb, sigma = model(pts, view_dirs)

        # Isolate the sampling dimension again, to perform volume rendering
        rgb = rgb.view(B, S, 3)               # [B, S, 3]
        sigma = sigma.squeeze(-1).view(B, S)  # [B, S]
        
        # Convert raw model outputs to RGB, depth, and accumulated opacity maps
        # [B, 3], [B], [B]
        rgb_map, depth_map, acc_map = volume_rendering(rgb, sigma, z_vals)

        # Calculate loss for this batch
        train_loss = criterion(rgb_map, target_rgb)

        # Backpropagate to compute the gradients
        train_loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Update the model parameters using the optimizer
        optimizer.step()
    
        # Log an image every `log_interval` iterations
        if i % config.log_interval == 0:
            # Run and log validation
            dataset.set_mode("val")
            model.eval()
            val_loss, rendered_img, target_img = validate(val_dataloader, model, criterion, config)

            # Log training and validation metrics
            wandb.log({"Learning Rate": lr})
            wandb.log({"Train Loss": train_loss.item()})
            wandb.log({"Validation Loss": val_loss.item()})
            # Log target and rendered image
            wandb.log({
                "Rendered Image": wandb.Image(rendered_img, 
                        caption=f"Rendered Image - Iter {i}"),
                "Target Image": wandb.Image(target_img, 
                        caption=f"Target Image - Iter {i}")
            })

            # Print the loss and save the model checkpoint
            print(f"\t\tIter {i}",
                  f"\nTrain Loss: {train_loss.item()}",
                  f"\nValidation Loss: {val_loss.item()}")
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
    run_id = "v0.0.2"
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
        "log_interval": 1        # Logging interval
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
    dataset = SceneDataset(images_dir=img_dir, 
                           poses_bounds_file=poses_path, 
                           S=config.num_samples, 
                           W=config.image_w,
                           H=config.image_h,
                           transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
    train(dataset, train_dataloader, val_dataloader, model, optimizer, config, save_path)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()