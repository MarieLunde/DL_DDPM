import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np



def get_dataloader_mnist(dataset_name, batch_size, train_or_test):

    # Define the transformation to scale the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Scale to the range [-1, 1]
    ])

    # Define the root directory where the dataset will be stored
    root_dir = './data' 

    dataset = torchvision.datasets.MNIST(root=root_dir, train=train_or_test, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
