import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
import matplotlib.pyplot as plt
from model import UNet



class DDPM(nn.Module):
  
    def __init__(self, beta_start = 0.0001, beta_end = 0.02, T = 1000, device='cpu'):
        super(DDPM, self).__init__()
        self.device = device
        self.T = T # Timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.beta_scheduler()  
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat((torch.tensor([1.], device=device), self.alpha_bar[:-1]))
        #assert self.alpha_bar_prev.shape == (self.T,)
    
    def beta_scheduler(self):     
        return torch.linspace(self.beta_start, self.beta_end, self.T, dtype = torch.float32, device=self.device) 

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
        model.eval()
        # sampeling initial gaussian noise, step 1 
        x = torch.randn((n_img, channels, img_shape, img_shape), device=device)
        for timestep in reversed(range(1, self.T)):  # step 2
            x = self.sampling_timestep_img(model, device, timestep, x)

        x0 = x
        return x0
      




