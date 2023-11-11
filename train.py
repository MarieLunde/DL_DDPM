import sys
import torch
from torch import nn
from model import Unet, DummyUnet


def train(epochs, dataset_name, device):
    """
    epochs: number of epochs
    dataset_name: 'MNIST' or 'CIFAR10
    device: 'cpu' or 'cuda'
    """
    data_loader = get_data_loader(dataset_name) #TODO (Marie): add data loader
    
    model = DummyUnet(image_size=28 if dataset_name == 'MNIST' else 256, 
                      channels= 1 if dataset_name == 'MNIST' else 3) #TODO (Anna): add real model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    MSE = nn.MSELoss()

    for i in epochs:

        # Algorithm 1 for a batch of images
        for images in data_loader:
            # line 2
            batch_size, channels, height, width = images.shape
            images = images.to(device)

            # line 3
            t = torch.zeros(batch_size) #TODO (Eline): from DiffusionModel

            # line 4
            epsilon = torch.zeros_like(images) #TODO (Eline): from DiffusionModel

            # line 5
            epsilon_theta = torch.zeros_like(images) #TODO (Eline):  from DiffusionModel
            loss = MSE(epsilon, epsilon_theta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #TODO: get metrics (FID, Inception score)
        #TODO (Anna): logging
        #TODO: save example images


if __name__ == '__main__':

    # Parse arguments
    if len (sys.argv) != 3: #checking if you have the right number of arguments
        print('Usage: python train.py <epochs> <dataset_name>')
        sys.exit(1)
    epochs = sys.argv[1]
    dataset_name = sys.argv[2]
    assert dataset_name in ['MNIST', 'CIFAR10']


    # Check if GPU is available
    USE_CUDA = torch.cuda.is_available()
    print("Running GPU.") if USE_CUDA else print("No GPU available.")
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # This is where the magic happens
    train(epochs=epochs, dataset_name=dataset_name, device=device)