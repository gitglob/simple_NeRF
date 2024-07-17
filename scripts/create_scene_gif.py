import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torchvision import transforms
from src.data import SceneDataset
from src.model import NeRF
from src.utils import tensor2PILimage, batch_encode, volume_rendering, create_gif


def infer(dataset, model):
    """ Run inference on the entire dataset with the pre-trained NeRF """    
    dataset.set_mode("val")
    model.eval()
    with torch.no_grad():
        rendered_imgs = []
        target_imgs = []
        for i in range(len(dataset)):
            print(f"Image: {i}/{len(dataset)}")
            # [W*H, 3], [W*H, S, 3], [W*H, S, 3], [W*H, S]
            target_rgb, pts, view_dirs, z_vals = dataset[i]
            target_rgb, pts, view_dirs, z_vals = target_rgb.cuda(), pts.cuda(), view_dirs.cuda(), z_vals.cuda()
            pts = batch_encode(pts, model.num_freqs_pos)             # [W*H, S, Dp]
            view_dirs = batch_encode(view_dirs, model.num_freqs_dir) # [W*H, S, Dd]

            # Split validation forward pass to batches so that it can fit to CUDA memory
            B = 512
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
                rgb_map_batch, _, _ = volume_rendering(rgb_batch, sigma_batch, z_vals_batch)
                
                # Insert batch results into preallocated tensors
                rendered_image[start:end] = rgb_map_batch  # [W*H, 3]
                target_image[start:end] = target_rgb_batch # [W*H, 3]

            # Convert tensors to images
            r = tensor2PILimage(rendered_image, 128, 128)
            t = tensor2PILimage(target_image, 128, 128)
            rendered_imgs.append(r)
            target_imgs.append(t)

    return rendered_imgs, target_imgs


def main():
    # Load images
    scene = "object"
    dataset_path = os.path.join(os.path.curdir, "data", "custom", scene)
    img_dir = os.path.join(dataset_path, "images")
    poses_path = os.path.join(dataset_path, "poses_bounds.npy")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = SceneDataset(images_dir=img_dir, 
                           poses_bounds_file=poses_path, 
                           S=64, W=128, H=128,
                           transform=transform)

    # Create model
    model = NeRF(10, 4).cuda()

    # Load model checkpoint
    checkpoint_pth = "checkpoints/model_object.pth"
    checkpoint = torch.load(checkpoint_pth)
    model.load_state_dict(checkpoint["model"])

    # Run inference on the entire dataset
    rendered_imgs, target_imgs = infer(dataset, model)

    # Create gifs for the rendered and target images
    create_gif(rendered_imgs, 'output/rendered.gif', duration=250)
    create_gif(target_imgs, 'output/target.gif', duration=250)

if __name__ == "__main__":
    main()