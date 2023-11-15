import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
def get_data(args):

    if args.dataset_path == "MNIST":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dataset = torchvision.datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms)


    elif args.dataset_path == "CIFAR10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root="CIFAR10", download=True, train=True, transform=transforms)

    elif args.dataset_path == "landscape_img_folder":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    else:
        raise AssertionError('Dataset not known!!')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader





def get_dataloader(dataset_name, batch_size):

    if dataset_name == "MNIST":
        # Define the transformation to scale the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Scale to the range [-1, 1]
            ])
        dataset = torchvision.datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
    elif dataset_name== "CIFAR10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = torchvision.datasets.CIFAR10(root="CIFAR10", download=True, train=True, transform=transforms)
    else:
        raise AssertionError('Unknown dataset')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def add_gaussian_noise(image, sigma): # TODO change noise function to our own
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise
    return noisy_image

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
train_loader = get_dataloader_mnist('MNIST', batch_size=1, train_or_test=True)

# Get the first batch
for images, labels in train_loader:
    break  # Exit the loop after the first batch

# Plot images with gradually added Gaussian noise
# plot_images_with_noise(images, labels)