import os
import numpy as np
import torch
from PIL import Image


"""
                            Complementary Use of SfM and NeRF

In practice, SfM and NeRF can be used together to leverage their respective strengths:
    SfM for Initialization: Use SfM to quickly estimate camera poses and a sparse point cloud, providing a good initialization for NeRF training.
                    - Sparse Representation: Lacks detailed information
                    - Does not provide texture or color
                    - No novel views
    NeRF for Refinement: Train NeRF using the poses from SfM to achieve a detailed and photorealistic representation of the scene.
                     - Dense Representation: Models the entire scene as a continuous volumetric field, capturing fine geometric and appearance details
                     - Provides accurate colors and through them, texture
                     - Provides novel views
                     - Implicit scene representation

Example Workflow
    Run SfM: Obtain camera poses and a sparse 3D point cloud.
    Prepare Data: Use the camera poses from SfM to sample rays for NeRF training.
    Train NeRF: Train the NeRF model using the sampled rays and the original images.
    Render Novel Views: Use the trained NeRF model to render high-quality novel views of the scene.
"""


class SceneDataset():
    def __init__(self, images_dir, poses_bounds_file, 
                 focal_length=24e-6, image_width=1536, 
                 sensor_width=9.8e-6,transform=None):
        """
        Args:
            images_dir (str): Directory with all the images.
            poses_bounds_file (str): File containing poses and bounds.
            focal_length (float): Focal length of the camera.
            image_width (int): The width of the dataset images (pixels)
            sensor_width (int): The sensor size for iPhone 15 pro (mm)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.focal_length_mm = focal_length
        self.focal_length_pixels = focal_length * image_width / sensor_width

        # Load poses and bounds
        poses_bounds = np.load(poses_bounds_file)

        # Extract poses of each camera (position and orientation in space) [R, t]
        self.poses = poses_bounds[:, :-5].reshape([-1, 3, 4])

        # Extract remaining column
        self.other = poses_bounds[:, -5:-2]

        # Extract bounds (near and far clipping planes) for each image
        self.bounds = poses_bounds[:, -2:]

        # Get list of image files
        self.image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.JPG')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to be fetched.
        
        Returns:
            tuple: (image, rays_o, rays_d, bounds)
                image (PIL Image): The image corresponding to the index [W*W, 3].
                rays_o (torch.Tensor): Ray origins [W*W, 3].
                rays_d (torch.Tensor): Ray directions [W*W, 3].
                bounds (np.array): The near and far bounds for this image.
        """
        idx = idx % len(self.image_files)
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).permute(1, 2, 0) # [W, H, 3]

        # Load pose and bounds
        pose = self.poses[idx]
        bounds = self.bounds[idx] # [2,]

        # Calculate rays
        H, W, C = image.shape
        image = image.view(-1, 3) # [W*H, 3]
        rays_o, rays_d = self.get_rays(H, W, pose) # [W*W, 3]
        bounds = torch.tensor(bounds, dtype=torch.float32).unsqueeze(0).repeat(W*W, 1) # [W*W, 2]

        return image.cuda(), rays_o.cuda(), rays_d.cuda(), bounds.cuda()

    def get_rays(self, image_height, image_width, camera_pose):
        """
        Generates camera rays for each pixel in the image.

        Pinhole camera model:
        https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html
        
        Args:
            image_height (int): Height of the image.
            image_width (int): Width of the image.
            camera_pose (np.array): 3x4 camera pose matrix.

        Returns:
            rays_o (torch.Tensor): Ray origins.
                                   This is the camera position in the world coordinate frame.
            rays_d (torch.Tensor): Ray directions.
                                   This is the unit vector (direction) from the camera center 
                                   to each pixel in the world coordinate frame.
        """
        # Create a mesh grid of image coordinates (pixel coordinates)
        i, j = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing='xy')
        
        # Center the x,y coordinates around the image center
        i = i - image_width * 0.5   # [W, W]
        j = j - image_height * 0.5  # [W, W]

        # Convert pixel coordinates to camera coordinates
        # dirs is essentially the direction vectors from the camera center to each pixel
        dirs = np.stack([i / self.focal_length_pixels,
                        -j / self.focal_length_pixels, 
                        -np.ones_like(i)], axis=-1)          # [W, W, 3]
        
        # Extract Rotation matrix and Translation vector
        camera_rot = camera_pose[:3, :3]   # [3, 3]
        camera_trans = camera_pose[:3, 3]  # [3, ]
        
        # Convert camera coordinates to normalized world coordinates
        rays_d = dirs @ camera_rot.T                              
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True) # [W, W, 3]
        
        # All rays originate from the camera origin
        rays_o = np.broadcast_to(camera_trans, rays_d.shape) # [W, W, 3]
        
        # Convert to torch tensors
        rays_o = torch.tensor(rays_o, dtype=torch.float32).view(-1, 3) # [W*W, 3]
        rays_d = torch.tensor(rays_d, dtype=torch.float32).view(-1, 3) # [W*W, 3]
        
        return rays_o, rays_d
