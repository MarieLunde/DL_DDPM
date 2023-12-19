import torchvision
import torch
from model import UNet
from ddpm import *
from PIL import Image


def load_model(dataset_name, device, dropout, learning_rate, path):
    """Loads a pickled model and optimizer from a given path"""
    channels = 1 if dataset_name == 'MNIST' else 3
    model = UNet(channels, channels, device = device, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    return model, optimizer, epoch, loss


def generate_images(image_shape, n_image_to_save, channels, device, model, dataset_name):
    """Generates images from a trained model and saves them to a folder"""
    output_folder = f'image_output_{dataset_name}'
    with torch.no_grad():
        generated_images = DDPM.sampling_image(image_shape, n_img = n_image_to_save, channels = channels, model = model, device = device)
    generated_images_numpy = generated_images.detach().cpu().numpy()

    # Save the images
    for i, image in enumerate(generated_images_numpy):
        torchvision.utils.save_image(torch.tensor(image), f"{output_folder}/genrated_img/{dataset_name}_{i}.png")


def save_imgs(images, file_path):
    """Saves a set of images to a file"""
    grid = torchvision.utils.make_grid(images)
    numpy_array = grid.permute(1, 2, 0).to('cpu').numpy()
    image = Image.fromarray(numpy_array)
    image.save(file_path)


if __name__=='__main__':
            
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    dropout =  0.1
    learning_rate =  0.001
    # Define the path to the saved model
    load_path = 'saved_models\CIFAR10_epoch0_model.pth' 
    model, optimizer, epoch, loss = load_model("CIFAR10", device, dropout, learning_rate, load_path)

    print(epoch)

