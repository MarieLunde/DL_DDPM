import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def get_dataloader(dataset_name, batch_size):
    dataset_path = '../mnist_data' if dataset_name == 'MNIST' else '../cifar10_data'

    dataloader = DataLoader(datasets.MNIST(dataset_path, 
                                download=True, 
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), # first, convert image to PyTorch tensor
                                    transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                ])), 
                batch_size=batch_size, 
                shuffle=True)
    
    return dataloader
    