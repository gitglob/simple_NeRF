import os
import itertools
import random
import wandb
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import tensor2image, save, load
from src.utils import batch_encode, volume_rendering


def validate(dataset, model, config):
    """
    Validate the NeRF.
    
    The idea here is that you run validation on all the rays of 1 image,
    but we still split them in batches to fit in my GPU memory.
    This is because during training we only get B rays from random images,
    but for validation we want ALL the rays from 1 random image, which
    results in W*H rays (much much bigger than B).
    """    
    with torch.no_grad():
        # Get a random image
        random_idx = random.randint(0, len(dataset) - 1)
        
        # [W*H, 3], [W*H, S, 3], [W*H, S, 3], [W*H, S]
        target_rgb, pts, view_dirs, z_vals = dataset[random_idx]
        target_rgb, pts, view_dirs, z_vals = target_rgb.cuda(), pts.cuda(), view_dirs.cuda(), z_vals.cuda()
        pts = batch_encode(pts, model.num_freqs_pos)             # [W*H, S, Dp]
        view_dirs = batch_encode(view_dirs, model.num_freqs_dir) # [W*H, S, Dd]

        # Split validation forward pass to batches so that it can fit to CUDA memory
        B = config.batch_size
        N, S = z_vals.shape
        rendered_image = torch.zeros((N, 3))  # [W*H, 3]
        target_image = torch.zeros((N, 3))    # [W*H, 3]
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
            
            # Insert batch results into preallocated tensors
            rendered_image[start:end] = rgb_map_batch  # [W*H, 3]
            target_image[start:end] = target_rgb_batch # [W*H, 3]
        
        # Calculate the validation loss
        loss = F.mse_loss(rendered_image, target_image).item()

        # Convert tensors to images
        rendered_image_np = tensor2image(rendered_image, config.image_w, config.image_h)
        target_image_np = tensor2image(target_image, config.image_w, config.image_h)

        return loss, rendered_image_np, target_image_np

def train(dataset, train_dataloader, 
          model, optimizer, config, save_path):
    """Train the NeRF."""
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Return to training mode
    dataset.set_mode("train")
    model.train()
    
    # Iterate over the epochs
    for i, batch in enumerate(itertools.cycle(train_dataloader)):
        # Parse B random rays sampled at S points
        # [B, 3], [B, S, 3], [B, S, 3], [B, S]
        target_rgb, pts, view_dirs, z_vals = batch
        target_rgb, pts, view_dirs, z_vals = target_rgb.cuda(), pts.cuda(), view_dirs.cuda(), z_vals.cuda()
        B, S, C = pts.shape

        # Exponentially decaying learning rate
        lr = config.lr_final + (config.lr - config.lr_final) * (1 - i / config.num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Apply positional encoding
        pts = batch_encode(pts, model.num_freqs_pos)              # [B, S, Dp] (Dp = 3 + 2*3*Fp)
        view_dirs = batch_encode(view_dirs, model.num_freqs_dir)  # [B, S, Dd] (Dd = 3 + 2*3*Fd)

        # Merge Batch and Sample dimensions
        pts = pts.view(B*S, -1)              # [B*S, Dp]
        view_dirs = view_dirs.view(B*S, -1)  # [B*S, Dd]

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
        
        # Update the model parameters using the optimizer
        optimizer.step()
    
        # Log an image every `log_interval` iterations
        if i % config.log_interval == 0:
            # Set training mode
            dataset.set_mode("val")
            model.eval()
            
            # Run and log validation
            val_loss, rendered_img, target_img = validate(dataset, model, config)

            # Log training and validation metrics
            wandb.log({"Learning Rate": lr})
            wandb.log({"Train Loss": train_loss.item()})
            wandb.log({"Validation Loss": val_loss})
            # Log target and rendered image
            wandb.log({
                "Rendered Image": wandb.Image(rendered_img),
                "Target Image": wandb.Image(target_img)
            })
            Image.fromarray(rendered_img).save(f"output/rendered/r_{i}.png")
            Image.fromarray(target_img).save(f"output/target/t_{i}.png")

            # Print the loss and save the model checkpoint
            print(f"\n\t\tIter {i}")
            print(f"Train Loss: {train_loss.item()}")
            print(f"Validation Loss: {val_loss}")
            save(model, optimizer, save_path)

            # Return to training mode
            dataset.set_mode("train")
            model.train()

        if i == config.num_iter:
            print("Iterations finished! Exiting loop...")
            break

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return
    
    # Dataset path
    scene = "object"
    dataset_path = os.path.join(os.path.curdir, "data", "custom", scene)

    # Initialize wandb
    project_name = f"simple_nerf-{scene}"

    # Initialize wandb
    run_id = "v1.0.0"
    wandb.init(project=project_name, 
            entity="gitglob", 
            resume='allow', 
            id=run_id)
    
    # Sweep parameters
    wandb.config.update({
        "num_iter": 100000,      # Number of epochs
        "lr": 5e-4,              # Learning rate start
        "lr_final": 5e-5,        # Learning rate finish
        "batch_size": 2048,      # Batch size (number of rays per iteration)
        "image_w": 128,          # Image weight dim
        "image_h": 128,          # Image height dim
        "num_samples": 64,       # Number of samples across each ray
        "num_freqs_pos": 10,     # Number of position encoding frequencies
        "num_freqs_dir": 4,      # Number of direction encoding frequencies
        "log_interval": 100      # Logging interval
    }, allow_val_change=True)
    config = wandb.config
    
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

    # Initialize model and optimizer
    model = NeRF(config.num_freqs_pos, config.num_freqs_dir).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/model_" + scene + ".pth"
    if os.path.exists(save_path):
        model, optimizer = load(model, optimizer, save_path)
        print(f"Model and optimizer loaded from {save_path}\n")
    else:
        print("No existing model...\n")

    # Train the model
    train(dataset, train_dataloader, 
          model, optimizer, config, save_path)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()