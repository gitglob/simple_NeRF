import sys
import os
from PIL import Image

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.utils import create_gif

def main():
    output_filename = 'output/timelapse.gif'
    folder_path = 'output/rendered'
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()
    images = [Image.open(os.path.join(folder_path, file)) for file in image_files]

    create_gif(images, output_filename, duration=250)

if __name__ == "__main__":
    main()