import numpy as np
import torch
import matplotlib.pyplot as plt
from model import UNet
from ddpm import *
import os
import torchvision



def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


# ddpm_instance = DDPM()  # You may need to pass any required parameters when creating an instance
# show_images(ddpm_instance.sampling_image(img_shape=[32,32], n_img = 1, channels = 1, model = UNet(1,1), device = None), f"Images generated")


  
def save_images(images, epoch, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, image in enumerate(images):
        torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/epoch{epoch}_sample{i}.png")

    print(f"Images saved at epoch {epoch}")   
    
     