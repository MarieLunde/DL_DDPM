import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
import math
#TODO: add utility class for DDPM

def beta_scheduler(self):    
    # Betas 
    return torch.linspace(self.beta_start, self.beta_end, self.T, dtype = torch.float64) 


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
          

    def sample_noise(x0):
        # torch.rand_like(something) = Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.     
        noise = torch.randn_like(x0)
        return x0 + noise    
    
    def noise_function(self, model, x0, noise, t):
        sqrt_alpha_bar_x0 = math.sqrt(self.alpha_bar)*x0
        sqrt_1_minus_alpha_bar_noise = math.sqrt(1-self.alpha_bar)*noise
        return model(sqrt_alpha_bar_x0 + sqrt_1_minus_alpha_bar_noise, t)    
        
    def sample_timestep(self, batchsize):
        # Sampling t from a uniform distribution
        # Batchsize is the batchsize of images, generating one t pr beta
        sampled_steps = torch.randint(1, self.T, (batchsize, )) 
        return sampled_steps      
    
    
    def sampling(self, img_shape, model, num_img, device):
        # sampeling inital gaussian noise
        x = torch.randn((num_img, img_shape[0], img_shape[1]),device=device)

        for timestep in reversed(range(self.T)):
            t = (torch.ones(num_img)*timestep).long().to(self.device)
            # sample random noise
            if timestep > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            pred_noise = model(x, t)
            var_t = (1 - self.alpha_bar_prev[timestep]) / (1 - self.alpha_bar[timestep]) * self.beta[timestep]
            model_mean = 1 / torch.sqrt(self.alpha[timestep]) * (x - ((1 - self.alpha[timestep]) / (torch.sqrt(1 - self.alpha_bar[timestep]))) * pred_noise)
        
        return model_mean + torch.sqrt(var_t) * z
    
    
       
        


 
        

