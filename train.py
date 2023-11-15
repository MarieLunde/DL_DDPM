import sys
import torch
from torch import nn
from model import UNet, DummyUnet
from dataloader import get_dataloader
from ddpm import DDPM


def train(dataset_name, epochs, batch_size, device):
    """
    dataset_name: 'MNIST' or 'CIFAR10
    epochs: number of epochs
    batch_size: batch size
    device: 'cpu' or 'cuda'
    """
    data_loader = get_dataloader(dataset_name, batch_size)
    
    model = DummyUnet(image_size=28 if dataset_name == 'MNIST' else 256, 
                      channels= 1 if dataset_name == 'MNIST' else 3) #TODO (Anna): add real model
    
    #model = UNet(1, 10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    MSE = nn.MSELoss()
    ddpm = DDPM()

    for epoch in range(epochs):
        print(epoch)

        # Algorithm 1 for a batch of images
        for images, labels in data_loader: # We don't actually use the labels
            # Algorithm 1, line 2
            images = images.to(device)

            # Algorithm 1, line 3
            #t = torch.randn(batch_size) #TODO (Eline): from DiffusionModel
            t = ddpm.sample_timestep(images.shape[0])

            # Algorithm 1, line 4
            #epsilon = torch.randn_like(images) #TODO (Eline): from DiffusionModel
            epsilon = ddpm.sample_noise(images)
            # Algorithm 1, line 5
            #epsilon_theta = torch.randn_like(images, requires_grad=True) #TODO (Eline):  from DiffusionModel, use t to get epsilon_theta
            epsilon_theta = ddpm.noise_function(model, images, epsilon, t)
            loss = MSE(epsilon, epsilon_theta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #TODO: get metrics (FID, Inception score)
        #TODO (Anna): logging
        #TODO: save example images


if __name__ == '__main__':

    # Parse arguments
    if len(sys.argv) != 4:
        print("Usage: python train.py <dataset_name> <epochs> <batch_size>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    assert dataset_name in ['MNIST', 'CIFAR10']
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])   


    # Check if GPU is available
    USE_CUDA = torch.cuda.is_available()
    print("Running GPU.") if USE_CUDA else print("No GPU available.")
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # This is where the magic happens
    train(dataset_name, epochs=epochs, batch_size=batch_size, device=device)

    