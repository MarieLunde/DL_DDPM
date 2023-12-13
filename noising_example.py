from ddpm import DDPM
from dataloader import get_dataloader
import torch
import torchvision
import os

USE_CUDA = torch.cuda.is_available()
print("Running GPU.") if USE_CUDA else print("No GPU available.")
device = 'cpu'
batch_size = 4
dataset_name = "MNIST"
dataloader = get_dataloader(dataset_name, batch_size)
ddpm = DDPM(device=device)
output_folder = "noising_example"
os.makedirs(output_folder, exist_ok=True)

print('noising')
images, label = next(iter(dataloader))
print(label)
print(images.shape)
x0 = images
for i in range(1, batch_size):
    x0[i] = images[0]

print(x0.shape)
#ts = ddpm.sample_timestep(batch_size)
ts = torch.tensor([0, 200, 400, 999])
noises = torch.randn_like(x0)
noised_imgs = ddpm._noise_function(noises, x0, ts)

for i, noised_img in enumerate(noised_imgs):
    torchvision.utils.save_image(noised_img, f"{output_folder}/noise_level_{ts[i].item()}.png")