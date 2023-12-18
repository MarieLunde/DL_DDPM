import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

num_workers = 4
def get_dataloader(dataset_name, batch_size):

    if dataset_name == "MNIST":
        # Define the transformation to scale the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to a PyTorch tensor
            transforms.Normalize((0.5,), (0.5,)),  # Scale to the range [-1, 1]
            transforms.Resize((32, 32))  # Stretch the i mage to 32x32
            ])
        dataset = datasets.MNIST(root=r"\data", download=True, train=True, transform=transform)
    elif dataset_name== "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),    # Randomly flip the image horizontally
            transforms.ToTensor(),
            transforms.Resize((32,32)),  # Stretch the image to 32x32
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=r"\data", download=True, train=True, transform=transform)
    else:
        raise AssertionError('Unknown dataset')
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader
