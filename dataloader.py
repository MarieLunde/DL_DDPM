import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

def get_dataloader(dataset_name, batch_size):

    if dataset_name == "MNIST":
        # Define the transformation to scale the MNIST dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Convert to a PyTorch tensor
            torchvision.transforms.Normalize((0.5,), (0.5,))  # Scale to the range [-1, 1]
            ])
        dataset = torchvision.datasets.MNIST(root="\data", download=True, train=True, transform=transform)
    elif dataset_name== "CIFAR10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = torchvision.datasets.CIFAR10(root="\data", download=True, train=True, transform=transform)
    else:
        raise AssertionError('Unknown dataset')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def add_gaussian_noise(image, sigma): # TODO change noise function to our own
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise
    return noisy_image



def noise_function(self, model, x0, noise, t):
    sqrt_alpha_bar_x0 = math.sqrt(self.alpha_bar)*x0
    sqrt_1_minus_alpha_bar_noise = math.sqrt(1-self.alpha_bar)*noise
    return model(sqrt_alpha_bar_x0 + sqrt_1_minus_alpha_bar_noise, t)  
    


def plot_images_with_noise(images, labels, num_images=10, sigma_range=(0, 1.5)):
    fig, axs = plt.subplots(1, num_images, figsize=(12, 6))
    fig.suptitle('Images with Gradually Added Gaussian Noise', fontsize=16)

    for i in range(num_images):
        # Original Image
        original_image = images.squeeze().numpy()
        axs[i].imshow(original_image, cmap='gray')
        axs[i].set_title(f'Label: {labels}')
        axs[i].axis('off')

        # Image with Gaussian Noise
        sigma = np.linspace(sigma_range[0], sigma_range[1], num_images)[i]
        noisy_image = add_gaussian_noise(images, sigma).squeeze().numpy()
        axs[i].imshow(noisy_image, cmap='gray')
        axs[i].set_title(f'Noise: {sigma:.2f}')
        axs[i].axis('off')

    plt.show()

# Get the dataloader for the training set
train_loader = get_dataloader('MNIST', batch_size=1)

# Get the first batch
for images, labels in train_loader:
    break  # Exit the loop after the first batch

# Plot images with gradually added Gaussian noise
# plot_images_with_noise(images, labels)