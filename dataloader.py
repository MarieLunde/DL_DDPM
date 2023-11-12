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

train_loader = get_dataloader_mnist('MNIST', batch_size=64, train_or_test=True)
test_loader = get_dataloader_mnist('MNIST', batch_size=64, train_or_test=False)




# Loop through the data loaders to check the images
for i, loader in enumerate([train_loader, test_loader]):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # Create a figure with 10 subplots

    for j in range(10):  # Display 10 images
        images, labels = next(iter(loader))  # Get a batch of images and labels

        # Display the images from the batch
        img = torchvision.utils.make_grid(images[j])
        img_np = img.numpy()  # Convert to a numpy array for visualization

        axes[j].imshow(np.transpose(img_np, (1, 2, 0)))
        axes[j].axis('off')
        if i == 0:  # For the first loop (training images)
            axes[j].set_title(f"Train Label: {labels[j]}")
        else:  # For the second loop (testing images)
            axes[j].set_title(f"Test Label: {labels[j]}")

    # plt.show()
