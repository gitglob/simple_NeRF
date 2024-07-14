import os
import itertools
import random
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import tensor2image, save, load
from src.utils import stratified_sampling, hierarchical_sampling, sample_rays_f, batch_encode, volume_rendering


def validate(dataset, model_c, model_f, config):
    """
    Validate the NeRF.
    
    The idea here is that you run validation on all the rays of 1 image,
    but we still split them in batches to fit in my GPU memory.
    This is because during training we only get B rays from random images,
    but for validation we want ALL the rays from 1 random image, which
    results in W*H rays (much much bigger than B).
    """
    with torch.no_grad():
        # Extract dimensions
        B = config.batch_size
        Nc = config.Nc
        Nf = config.Nf
        fp = config.fp
        fd = config.fd
       
        # Get a random image
        random_idx = random.randint(0, len(dataset) - 1)
        
        # Parse the rays of 1 random image
        # [W*H, 3], [W*H, Nc, 3], [W*H, Nc, 3], [W*H, Nc]
        target_rgb, rays_o, rays_d, bounds = dataset[random_idx]
        target_rgb, rays_o, rays_d, bounds = target_rgb.cuda(), rays_o.cuda(), rays_d.cuda(), bounds.cuda()

        # Split validation forward pass to batches so that it can fit to CUDA memory
        M = rays_o.shape[0]
        rendered_image_c = torch.zeros((M, 3))  # [W*H, 3]
        rendered_image_cf = torch.zeros((M, 3)) # [W*H, 3]
        target_image = torch.zeros((M, 3))      # [W*H, 3]
        for start in range(0, M, B):
            # Get the batch tensors
            end = start + B
            b_rays_o = rays_o[start:end]         # [B, Nc, Dp]
            b_rays_d = rays_d[start:end]         # [B, Nc, Dd]
            b_bounds = bounds[start:end]         # [B, Nc]
            b_target_rgb = target_rgb[start:end] # [B, 3]

            # Sample the rays at Nc points
            # [W*H, Nc, 3], [W*H, Nc, 3], [W*H, Nc]
            pts_c, view_dirs_c, z_vals_c = stratified_sampling(b_rays_o, b_rays_d, b_bounds, Nc)
            
            # Encode point and direction rays
            pts_enc_c = batch_encode(pts_c, fp)             # [W*H, Nc, Dp]
            view_dirs_enc_c = batch_encode(view_dirs_c, fd) # [W*H, Nc, Dd]

            # Merge Batch and Sample dimensions
            pts_enc_c = pts_enc_c.view(B*Nc, -1)              # [B*Nc, Dp]
            view_dirs_enc_c = view_dirs_enc_c.view(B*Nc, -1)  # [B*Nc, Dd]

            # Query the coarse model
            # [B*Nc, 3], [B*Nc, 1]
            rgb_c, sigma_c = model_c(pts_enc_c, view_dirs_enc_c)
            
            # Isolate the sampling dimension again, to perform volume rendering
            rgb_c = rgb_c.view(B, Nc, 3)               # [B, Nc, 3]
            sigma_c = sigma_c.squeeze(-1).view(B, Nc)  # [B, Nc]

            # Perform hierarchical sampling to get the fine Z values
            z_vals_f = hierarchical_sampling(rgb_c, sigma_c, z_vals_c, Nf) # [B, Nf, 3]
            z_vals_cf = torch.cat((z_vals_c, z_vals_f), dim=1)             # [B, Nc+Nf, 3]

            # Sample rays for the fine network
            # [B, Nf, 3], [B, Nf, 3]
            pts_f, view_dirs_f = sample_rays_f(b_rays_o, b_rays_d, z_vals_f, Nf)

            # Apply positional encoding
            pts_enc_f = batch_encode(pts_f, fp)              # [B, Nf, Dp]
            view_dirs_enc_f = batch_encode(view_dirs_f, fd)  # [B, Nf, Dd]

            # Merge Batch and Sample dimensions
            pts_enc_f = pts_enc_f.view(B*Nf, -1)              # [B*Nf, Dp]
            view_dirs_enc_f = view_dirs_enc_f.view(B*Nf, -1)  # [B*Nf, Dd]

            # Merge the coarse and fine batch
            pts_enc_cf = torch.cat((pts_enc_c, pts_enc_f), dim=0)                   # [B*(Nc+Nf), Dp]
            view_dirs_enc_cf = torch.cat((view_dirs_enc_c, view_dirs_enc_f), dim=0) # [B*(Nc+Nf), Dd]

            # Query the fine model
            # [B*(Nc+Nf), 3], [B*(Nc+Nf), 1]
            rgb_map_cf, sigma_cf = model_f(pts_enc_cf, view_dirs_enc_cf)

            # Isolate the sampling dimension again, to perform volume rendering
            rgb_map_cf = rgb_map_cf.view(B, Nc+Nf, 3)       # [B, Nc+Nf, 3]
            sigma_cf = sigma_cf.squeeze(-1).view(B, Nc+Nf)  # [B, Nc+Nf]
            
            # Convert raw model outputs to RGB pixels and new Z samples for the fine network
            # [B, 3]
            rgb_map_c = volume_rendering(rgb_c, sigma_c, z_vals_c)
            # [B, 3]
            rgb_map_cf = volume_rendering(rgb_map_cf, sigma_cf, z_vals_cf)
            
            # Insert batch results into preallocated tensors
            rendered_image_c[start:end] = rgb_map_c   # [W*H, 3]
            rendered_image_cf[start:end] = rgb_map_cf # [W*H, 3]
            target_image[start:end] = b_target_rgb    # [W*H, 3]
        
        # Calculate the validation loss
        loss = torch.sum((rendered_image_c - target_image) ** 2  + (rendered_image_cf - target_image) ** 2).item()

        # Convert tensors to images
        rendered_image_np_c = tensor2image(rendered_image_c, config.image_w, config.image_h)
        rendered_image_np_f = tensor2image(rendered_image_cf, config.image_w, config.image_h)
        target_image_np = tensor2image(target_image, config.image_w, config.image_h)

        return loss, rendered_image_np_c, rendered_image_np_f, target_image_np

def train(dataset, train_dataloader, 
          model_c, model_f,
          optimizer, config, save_path):
    """Train the NeRF."""    
    # Return to training mode
    dataset.set_mode("train")
    model_c.train()
    model_f.train()
    
    # Extract dimension sizes
    B = config.batch_size
    Nc = config.Nc
    Nf = config.Nf
    fp = config.fp
    fd = config.fd

    # Iterate over the epochs
    for i, batch in enumerate(itertools.cycle(train_dataloader)):
        # Exponentially decaying learning rate
        lr = config.lr_final + (config.lr - config.lr_final) * (1 - i / config.num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Parse B random rays
        # [B, 3], [B, 3], [B, 3], [B, 2]
        target_rgb, rays_o, rays_d, bounds = batch
        target_rgb, rays_o, rays_d, bounds = target_rgb.cuda(), rays_o.cuda(), rays_d.cuda(), bounds.cuda()

        # Sample the rays at Nc points
        # [B, Nc, 3], [B, Nc, 3], [B, Nc]
        pts_c, view_dirs_c, z_vals_c = stratified_sampling(rays_o, rays_d, bounds, Nc)

        # Apply sin-cos encoding
        pts_enc_c = batch_encode(pts_c, fp)             # [B, Nc, Dp]
        view_dirs_enc_c = batch_encode(view_dirs_c, fd) # [B, Nc, Dd]

        # Merge Batch and Sample dimensions
        pts_enc_c = pts_enc_c.view(B*Nc, -1)             # [B*Nc, Dp]
        view_dirs_enc_c = view_dirs_enc_c.view(B*Nc, -1) # [B*Nc, Dd]

        # Set optimizer gradient to 0
        optimizer.zero_grad()

        # Query the model with the sampled points
        # [B*Nc, 3], [B*Nc, 1]
        rgb_c, sigma_c = model_c(pts_enc_c, view_dirs_enc_c)

        # Isolate the sampling dimension again, to perform volume rendering
        rgb_c = rgb_c.view(B, Nc, 3)               # [B, Nc, 3]
        sigma_c = sigma_c.squeeze(-1).view(B, Nc)  # [B, Nc]
        
        # Convert raw model outputs to new Z coordinates to sample for the fine network
        # [B, Nf]
        z_vals_f = hierarchical_sampling(rgb_c, sigma_c, z_vals_c, Nf=Nf)
        z_vals_cf = torch.cat((z_vals_c, z_vals_f), dim=1)

        # Sample rays for the fine network
        # [B, Nf, 3], [B, Nf, 3]
        pts_f, view_dirs_f = sample_rays_f(rays_o, rays_d, z_vals_f, Nf)

        # Apply sin-cos encoding
        pts_enc_f = batch_encode(pts_f, fp)              # [B, Nf, Dp]
        view_dirs_enc_f = batch_encode(view_dirs_f, fd)  # [B, Nf, Dd]

        # Merge Batch and Sample dimensions
        pts_enc_f = pts_enc_f.view(B*Nf, -1)              # [B*Nf, Dp]
        view_dirs_enc_f = view_dirs_enc_f.view(B*Nf, -1)  # [B*Nf, Dd]

        # Merge the coarse and fine batch
        pts_enc_cf = torch.cat((pts_enc_c, pts_enc_f), dim=0)                   # [B*(Nc+Nf), Dp]
        view_dirs_enc_cf = torch.cat((view_dirs_enc_c, view_dirs_enc_f), dim=0) # [B*(Nc+Nf), Dd]

        # Query the model with the sampled points
        # [B*(Nc+Nf), 3], [B*(Nc+Nf), 1]
        rgb_cf, sigma_cf = model_f(pts_enc_cf, view_dirs_enc_cf)

        # Isolate the sampling dimension again, to perform volume rendering
        rgb_cf = rgb_cf.view(B, Nc+Nf, 3)               # [B, Nc+Nf, 3]
        sigma_cf = sigma_cf.squeeze(-1).view(B, Nc+Nf)  # [B, Nc+Nf]
        
        # Convert raw model outputs to RGB pixels and new Z samples for the fine network
        # [B, 3]
        rgb_map_c = volume_rendering(rgb_c, sigma_c, z_vals_c)
        rgb_map_cf = volume_rendering(rgb_cf, sigma_cf, z_vals_cf)

        # Calculate loss for this batch
        train_loss = torch.sum( (rgb_map_c - target_rgb) ** 2  + (rgb_map_cf - target_rgb) ** 2 )

        # Backpropagate to compute the gradients
        train_loss.backward()
        
        # Update the model parameters using the optimizer
        optimizer.step()
    
        # Log an image every `log_interval` iterations
        if i % config.log_interval == 0:
            # Set training mode
            dataset.set_mode("val")
            model_c.eval()
            model_f.eval()
            
            # Run and log validation
            val_loss, rendered_img_c, rendered_img_f, target_img = validate(dataset, model_c, model_f, config)

            # Log training and validation metrics
            wandb.log({"Learning Rate": lr})
            wandb.log({"Train Loss": train_loss.item()})
            wandb.log({"Validation Loss": val_loss})
            # Log target and rendered image
            wandb.log({
                "Coarse Rendered Image": wandb.Image(rendered_img_c, 
                        caption=f"Fine Image Iter {i}"),
                "Fine Rendered Image": wandb.Image(rendered_img_f, 
                        caption=f"Coarse Image - Iter {i}"),
                "Target Image": wandb.Image(target_img, 
                        caption=f"Target Image - Iter {i}")
            })

            # Print the loss and save the model checkpoint
            print(f"\n\t\tIter {i}")
            print(f"Train Loss: {train_loss.item()}")
            print(f"Validation Loss: {val_loss}")
            save(model_c, model_f, optimizer, save_path)

            # Return to training mode
            dataset.set_mode("train")
            model_c.train()
            model_f.train()

        if i == config.num_iter:
            print("Iterations finished! Exiting loop...")
            break

def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return
    device = torch.device("cuda")
    
    # Dataset path
    scene = "trex_fc"
    dataset_path = os.path.join(os.path.curdir, "data", "NeRF", "nerf_llff_data", "trex")

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
        "num_iter": 50000,   # Number of epochs
        "lr": 5e-4,          # Learning rate start
        "lr_final": 5e-5,    # Learning rate finish
        "batch_size": 512,   # Batch size (number of rays per iteration)
        "Nc": 64,            # Coarse sampling
        "Nf": 128,           # Fine sampling
        "image_w": 128,      # Image weight dim
        "image_h": 128,      # Image height dim
        "fp": 10,            # Number of position encoding frequencies
        "fd": 4,             # Number of direction encoding frequencies
        "log_interval": 100  # Logging interval
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
                           W=config.image_w,
                           H=config.image_h,
                           transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model and optimizer
    model_c = NeRF(config.fp, config.fd, device=device).cuda()
    model_f = NeRF(config.fp, config.fd, device=device).cuda()
    params = list(model_c.parameters()) + list(model_f.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/model_" + scene + ".pth"
    if os.path.exists(save_path):
        model_c, model_f, optimizer = load(model_c, model_f, optimizer, save_path)
        print(f"Model and optimizer loaded from {save_path}\n")
    else:
        print("No existing model...\n")

    # Train the model
    train(dataset, train_dataloader, 
          model_c, model_f, 
          optimizer, config, save_path)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()