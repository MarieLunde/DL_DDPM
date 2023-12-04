from utils import load_model
import torch

dataset_name='MNIST'
model_path = f'saved_models/{dataset_name}.pth'

USE_CUDA = torch.cuda.is_available()
print("Running GPU.") if USE_CUDA else print("No GPU available.")
device = torch.device("cuda" if USE_CUDA else "cpu")

# we're not going to train here so it doesn't matter
learning_rate = 1
dropout = 1

print('loading model')

model = load_model(dataset_name, device, dropout, learning_rate, path=model_path)
print('done')