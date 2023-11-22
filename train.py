import sys
import torch
from torch import nn
from model import UNet
from dataloader import get_dataloader
from ddpm import DDPM
from utils import *
try:
    import wandb
    with_logging = False
except:
    print("Wandb not installed. Logging will not work.")
    with_logging = False

save_images = True


def train(dataset_name, epochs, batch_size, device):
    """
    dataset_name: 'MNIST' or 'CIFAR10
    epochs: number of epochs
    batch_size: batch size
    device: 'cpu' or 'cuda'
    """
    data_loader = get_dataloader(dataset_name, batch_size)
    
    #model = DummyUnet(image_size=28 if dataset_name == 'MNIST' else 256, 
    #                  channels= 1 if dataset_name == 'MNIST' else 3)
    channels = 1 if dataset_name == 'MNIST' else 3
    image_shape = 32 
    model = UNet(channels, channels)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    MSE = nn.MSELoss()
    ddpm = DDPM(device=device)
    ddpm.to(device)
    save_interval = 2  # Save images every second epoch
    output_folder = f'image_output_{dataset_name}'


    for epoch in range(epochs):
        model.train()
        print(epoch)

        # Algorithm 1 for a batch of images
        i = 0 #TO REMOVE
        for images, labels in data_loader: # We don't actually use the labels
            # Algorithm 1, line 2
            images = images.to(device)

            # Algorithm 1, line 3
            t = ddpm.sample_timestep(images.shape[0])

            # Algorithm 1, line 4
            epsilon = ddpm.sample_noise(images)

            # Algorithm 1, line 5
            epsilon = torch.randn_like(images)
            epsilon_theta = ddpm.noise_function(model, images, epsilon, t)
            loss = MSE(epsilon, epsilon_theta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss (batch)", loss)
            i += 1 #TO REMOVE
            if i == 10: #TO REMOVE
                break #TO REMOVE
        print("Loss (epoch)", loss)

        #TODO (Eline): get metrics (FID, Inception score)
        if with_logging:
            wandb.log({"loss": loss,
                    "FID": 0 #TODO (Eline): replace 0 with FID score
                    })
        
        #TODO (Marie): save example images
        if epoch % save_interval == 0 and save_images == True:
            print("sampleing")
            with torch.no_grad():
                generated_images = ddpm.sampling_image(image_shape, n_img = 1, channels = channels, model = model, device = device)

            generated_images_numpy = generated_images.detach().cpu().numpy()

            # Save the images
            for i, image in enumerate(generated_images_numpy):
                torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/epoch{epoch}_sample{i}.png")


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

    # Initialize logging
    if with_logging:
            
        wandb.init(
        project="diffusion-project", entity="team-perfect-pitch",
        config={
        "device": device,
        "architecture": "UNet",
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        }
    )

    # This is where the magic happens
    train(dataset_name, epochs=epochs, batch_size=batch_size, device=device)

