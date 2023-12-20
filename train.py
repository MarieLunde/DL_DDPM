import sys
import os
import torch
from torch import nn
from model import UNet
from dataloader import get_dataloader
from ddpm import DDPM
import wandb
from metrics import preprocess_fid_score, inception_score
from torchmetrics.image.fid import FrechetInceptionDistance
import time 
import datetime
import torchvision
from utils import *
from torch.optim.lr_scheduler import CyclicLR

with_logging = True
save_images = True
n_image_to_save = 6 # Number of images saved every xx epoch
save_model = True
save_interval = 10  # Save images every xx epoch
save_metrics = True


def train(dataset_name, epochs, batch_size, device, dropout, learning_rate, gradient_clipping):
    """
    dataset_name: 'MNIST' or 'CIFAR10
    epochs: number of epochs
    batch_size: batch size
    device: 'cpu' or 'cuda'
    """
    data_loader = get_dataloader(dataset_name, batch_size)
    
    channels = 1 if dataset_name == 'MNIST' else 3
    image_shape = 32 
    model = UNet(channels, channels, device = device, dropout=dropout)

    print("model params", next(model.parameters()).get_device())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Specify cyclic learning rate parameters
    base_lr = 0.001  # Initial learning rate
    max_lr = 0.01    # Maximum learning rate
    step_size_up = 100  # Number of steps for the learning rate to increase
    step_size_down = 100  # Number of steps for the learning rate to decrease

    # Create a cyclic learning rate scheduler
    scheduler = CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular')


    MSE = nn.MSELoss()
    ddpm = DDPM(device=device)
    ddpm.to(device)
    fid_dim = 64
    fid = FrechetInceptionDistance(feature=fid_dim, reset_real_features=False)
    fidscore = None # in case there is no metrics saving, but there is logging
    
    if save_images:
        output_folder_root = f'image_output_{dataset_name}'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_folder = os.path.join(output_folder_root, f"run_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)


    model.train()

    for epoch in range(epochs):
        print(epoch)

        # Algorithm 1 for a batch of images
        for images, _ in data_loader: # We don't actually use the labels
            # Algorithm 1, line 2
            images = images.to(device)

            # Algorithm 1, line 3
            current_batch_size = images.shape[0] # truncated on last epoch
            
            # Algorithm 1, line 4 and 5
            epsilon_theta, epsilon = ddpm.noising_function(images, current_batch_size, model)
            loss = MSE(epsilon, epsilon_theta)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scheduler.step()

            # we only compute real features in the first batch to save time
            if save_metrics and epoch == 0:
                images_unnormalized = ((images.clamp(-1, 1) + 1) / 2)*255
                fid.update(preprocess_fid_score(images_unnormalized), real=True)


        print("Loss (epoch)", loss)

        if save_metrics and epoch % save_interval == 0:
            print('saving metrics -allow min 10 minutes')
            print('gen', time.strftime("%H:%M:%S", time.localtime()))
            n_image_to_gen = 64 # we can't load all in at the same time
            for i in range((fid_dim // n_image_to_gen)+1): 
                generated_images = ddpm.sampling_image(model= model, num_img = n_image_to_save, channels = channels, img_shape=image_shape)
                fid.update(preprocess_fid_score(generated_images), real=False)
            fidscore = fid.compute() # this also resets fid
            print("FID", fidscore)

        
        if save_images and epoch % save_interval == 0:
            print("sampleing")
            generated_images = ddpm.sampling_image(model= model, num_img = n_image_to_save, channels = channels, img_shape=image_shape)
            save_imgs(generated_images, f"{output_folder}/epoch{epoch}.jpg")

        #diversity, quality = inception_score(generated_images)
        diversity, quality = 0, 0
        print("inception score", diversity, quality)
        if with_logging:
            wandb.log({"loss": loss,
                    "FID": fidscore,
                    "Inception (diversity)": diversity,
                    "Inception (quality)": quality
                    })
        
        if save_model and epoch % 10 == 0:
             save_directory = f'saved_model_{wandb.run.name}'
             # Check if the directory exists, and if not, create it
             if not os.path.exists(save_directory):
                 os.makedirs(save_directory)

            # Construct the save path with model architecture and epoch information
             save_path = os.path.join(save_directory, f'{dataset_name}.pth')            
             # Save the trained model to the specific directory
             torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss
             }, save_path)



if __name__ == '__main__':

    # Parse arguments
    if len(sys.argv) < 4:
        print("Usage: python train.py <dataset_name> <epochs> <batch_size> <dropout> <learning_rate>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    assert dataset_name in ['MNIST', 'CIFAR10']
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])   
    dropout = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.1
    learning_rate = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.001
    gradient_clipping = bool(sys.argv[6]) if len(sys.argv) >= 7 else True


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
        "dropout": dropout,
        "learning_rate": learning_rate,
        "gradient_clipping": gradient_clipping
        }
    )
        
    # This is where the magic happens
    train(dataset_name, epochs=epochs, batch_size=batch_size, device=device, 
          dropout=dropout, learning_rate=learning_rate, gradient_clipping=gradient_clipping)

