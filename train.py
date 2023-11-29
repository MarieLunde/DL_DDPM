import sys
import os
import torch
from torch import nn
from model import UNet
from dataloader import get_dataloader
from ddpm import DDPM
from utils import *
import wandb


with_logging = True
save_images = True
save_model = True


def train(dataset_name, epochs, batch_size, device, dropout):
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
    model = UNet(channels, channels, device = device, dropout=dropout)

    print("model params", next(model.parameters()).get_device())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    MSE = nn.MSELoss()
    ddpm = DDPM(device=device)
    ddpm.to(device)
    
    if save_images:
        save_interval = 20  # Save images every second epoch
        output_folder_root = f'image_output_{dataset_name}'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_folder = os.path.join(output_folder_root, f"run_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)


    for epoch in range(epochs):
        model.train()
        print(epoch)

        # Algorithm 1 for a batch of images
        i = 0 #TO REMOVE
        for images, labels in data_loader: # We don't actually use the labels
            # Algorithm 1, line 2
            images = images.to(device)

            # Algorithm 1, line 3
            t = ddpm.sample_timestep(images.shape[0]).to(device)

            # Algorithm 1, line 4 and 5
            epsilon_theta, epsilon = ddpm.noise_function(model, images, t)
            loss = MSE(epsilon, epsilon_theta)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()


            print("Loss (batch)", loss)
            i += 1 #TO REMOVE
            if i == 10: #TO REMOVE
                break #TO REMOVE
        print("Loss (epoch)", loss)
        
        if epoch % save_interval == 0 and save_images:
            print("sampleing")
            with torch.no_grad():
                generated_images = ddpm.sampling_image(image_shape, n_img = 2, channels = channels, model = model, device = device)

            generated_images_numpy = generated_images.detach().cpu().numpy()

            # Save the images
            for i, image in enumerate(generated_images_numpy):
                torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/epoch{epoch}_sample{i+1}.png")

        fidscore = fid_score(images, generated_images, device)
        inception_score = inception(generated_images, inception_model, device)
        if with_logging:
            wandb.log({"loss": loss,
                    "FID": fidscore
                    })
        
        if epoch % save_interval == 0 and save_model:
            save_directory = 'saved_models'

            # Check if the directory exists, and if not, create it
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Save the trained model to a specific directory
            save_path = f'saved_models/{dataset_name}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)


if __name__ == '__main__':

    # Parse arguments
    if len(sys.argv) < 4:
        print("Usage: python train.py <dataset_name> <epochs> <batch_size>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    assert dataset_name in ['MNIST', 'CIFAR10']
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])   
    dropout = float(sys.argv[4]) if len(sys.argv) == 5 else 0.4


    # Check if GPU is available
    USE_CUDA = torch.cuda.is_available()
    print("Running GPU.") if USE_CUDA else print("No GPU available.")
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Initialize logging
    if with_logging:
        print("with logging")
            
        wandb.init(
        project="diffusion-project", entity="team-perfect-pitch",
        config={
        "device": device,
        "architecture": "UNet",
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout
        }
    )

    # This is where the magic happens
    train(dataset_name, epochs=epochs, batch_size=batch_size, device=device, dropout=dropout)

