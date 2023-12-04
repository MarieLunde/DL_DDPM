from utils import load_model
from dataloader import get_dataloader
import torch
from metrics import preprocess_fid_score
from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm import DDPM
import datetime
import torchvision
import os

output_folder = 'final_samples'
os.makedirs(output_folder, exist_ok=True)

batch_size = 32

dataset_name='MNIST'
model_path = f'saved_models/{dataset_name}.pth'

USE_CUDA = torch.cuda.is_available()
print("Running GPU.") if USE_CUDA else print("No GPU available.")
device = torch.device("cuda" if USE_CUDA else "cpu")

# we're not going to train here so it doesn't matter
learning_rate = 1
dropout = 1

print('getting data loader')

data_loader = get_dataloader(dataset_name, batch_size)

print('loading model')
model, _, _, _ = load_model(dataset_name, device, dropout, learning_rate, path=model_path)


fid_dim = 2048
fid = FrechetInceptionDistance(feature=fid_dim, reset_real_features=False)
fid.to(device)

print(fid._device)

ddpm = DDPM(device=device)

print('getting cov and mean for real images')
i = 0
for images, _ in data_loader:
    images = images.to(device)
    images_unnormalized = ((images.clamp(-1, 1) + 1) / 2)*255
    fid.update(preprocess_fid_score(images_unnormalized, device), real=True)
    i += 1
    if i%1000 == 0:
        print(i, "batches processed")

_, channels, _, image_shape = images.shape


print('getting cov and mean for generated images')
num_fids_for_confint = 10
fids = []
for k in range(num_fids_for_confint):

    for i in range((fid_dim // batch_size)+1): 
        print(i)
        if i%100 == 0:
            print(i)
        with torch.no_grad():
            generated_images = ddpm.sampling_image(image_shape, n_img = batch_size, channels = channels, model = model, device = device)

        
        fid.update(preprocess_fid_score(generated_images, device), real=False)

        if k == 0: # saving images for poster
            generated_images_numpy = generated_images.detach().cpu().numpy()

            # Save the images
            for j, image in enumerate(generated_images_numpy):
                torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/round{i}_sample{j+1}.png")

    fidscore = fid.compute()
    print(fidscore)
    fids.append(fidscore.item())
    with open("final_fids.txt", 'w') as outfile:
        outfile.write(fids)


