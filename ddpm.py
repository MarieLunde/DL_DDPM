import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
import matplotlib.pyplot as plt
from model import UNet

#TODO: add utility class for DDPM

def beta_scheduler(self):    
    # Betas 
    return torch.linspace(self.beta_start, self.beta_end, self.T, dtype = torch.float32) 


class DDPM:
  
    def __init__(self, beta_start = 0.0001, beta_end = 0.02, T = 1000):
        
        self.T = T # Timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_scheduler(self)  
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = np.append(1., self.alpha_bar[:-1])
        assert self.alpha_bar_prev.shape == (self.T,)
          

    def sample_noise(self, x0):
        # torch.rand_like(something) = Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.     
        noise = torch.randn_like(x0)
        return noise    
    
    def noise_function(self, model, x0, noise, t):
        sqrt_alpha_bar_x0 = torch.sqrt(self.alpha_bar[t][:,None, None, None])*x0
        sqrt_1_minus_alpha_bar_noise = torch.sqrt(1-self.alpha_bar[t][:,None, None, None])*noise
        noised_img = sqrt_alpha_bar_x0 + sqrt_1_minus_alpha_bar_noise
        return model(noised_img, t)
        
    def sample_timestep(self, batchsize):
        # Sampling t from a uniform distribution
        # Batchsize is the batchsize of images, generating one t pr beta
        sampled_steps = torch.randint(1, self.T, (batchsize, )) 
        return sampled_steps    
    
    
    def sampling_timestep_img(self, model, device, timestep, x):
        t = torch.tensor(timestep, dtype=torch.long, device=device)
        print(t)
        # sample random noise, step 3
        if timestep > 0:
            z = torch.randn_like(x)
        else:
            z = 0
        # step 4
        pred_noise = model(x, t)
        var_t = (1 - self.alpha_bar_prev[timestep]) / (1 - self.alpha_bar[timestep]) * self.beta[timestep]
        model_mean = 1 / torch.sqrt(self.alpha[timestep]) * (x - ((1 - self.alpha[timestep]) / (torch.sqrt(1 - self.alpha_bar[timestep]))) * pred_noise)
    
        return model_mean + torch.sqrt(var_t) * z

    def sampling_image(self, img_shape, n_img, channels, model, device):
        # sampeling initial gaussian noise, step 1 
        x = torch.randn((n_img, channels, img_shape[0], img_shape[1]), device=device)
        print(x)
        for timestep in reversed(range(1, 10)):  # step 2
            x = DDPM.sampling_timestep_img(self, model, device, timestep, x)

        x0 = x
        print(x0)
        return x0




    def show_images(images, title=""):
        """Shows the provided images as sub-pictures in a square"""

        # Converting images to CPU numpy arrays
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx][0], cmap="gray")
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Showing the figure
        plt.show()
        

ddpm_instance = DDPM()  # You may need to pass any required parameters when creating an instance
# DDPM.show_images(ddpm_instance.sampling_image(img_shape=[32,32], n_img = 1, channels = 1, model = UNet(1,1), device = None), f"Images generated")
 
        




