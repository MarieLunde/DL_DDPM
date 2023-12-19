from utils import load_model
from dataloader import get_dataloader
import torch
from metrics import preprocess_fid_score
from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm import DDPM
import datetime
import torchvision
import pickle
import os
save_images = False
dataset_name='MNIST'
output_folder = f'final_samples_{dataset_name}'
os.makedirs(output_folder, exist_ok=True)

batch_size = 32


USE_CUDA = torch.cuda.is_available()
print("Running GPU.") if USE_CUDA else print("No GPU available.")
device = torch.device("cuda" if USE_CUDA else "cpu")

# we're not going to train here so it doesn't matter
learning_rate = 1
dropout = 1

print('getting data loader')

data_loader = get_dataloader(dataset_name, batch_size)

fid_dim = 2048
fid = FrechetInceptionDistance(feature=fid_dim, reset_real_features=False)
fid.to(device)

print(fid._device)

ddpm = DDPM(device=device)

print('getting cov and mean for real images')
pickle_name = f'fid_{dataset_name}_{fid_dim}.pkl'
if not pickle_name in os.listdir():
    i = 0
    for images, _ in data_loader:
        images = images.to(device)
        images_unnormalized = ((images.clamp(-1, 1) + 1) / 2)*255
        fid.update(preprocess_fid_score(images_unnormalized, device), real=True)
        i += 1
        if i%100 == 0:
            print(i, "batches processed")

    with open(pickle_name, 'wb') as f:
        pickle.dump(fid, f)
        print('pickled')
else:
    print('loading from file')
    with open(pickle_name, 'rb') as f:
        fid = pickle.load(f)
    print('loaded')

channels = 3 if dataset_name == 'CIFAR10' else 1
image_shape = 32

print('getting cov and mean for generated images')
model_paths = [f'saved_models/{dataset_name}_{i}.pth' for i in range(4, 7)]
fids = []
for k, model_path in enumerate(model_paths):

    print('loading model', model_path)
    model, _, _, _ = load_model(dataset_name, device, dropout, learning_rate, path=model_path)

    for i in range((fid_dim // batch_size)+1): 
        if i%100 == 0:
            print(i)
        with torch.no_grad():
            generated_images = ddpm.sampling_image(img_shape=image_shape, model= model, num_img = batch_size, channels = channels)
        fid.update(preprocess_fid_score(generated_images, device), real=False)

        if save_images and k == 0: # saving images for poster
            generated_images_numpy = generated_images.detach().cpu().numpy()

            # Save the images
            for j, image in enumerate(generated_images_numpy):
                torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/round{i}_sample{j+1}.png")

    fidscore = fid.compute()
    print(fidscore)
    fids.append(str(fidscore.item()))
    print(fids)
    #with open(f"final_fids_{fid_dim}_{dataset_name}.txt", 'w') as outfile:
    #    outfile.write(' '.join(fids))


