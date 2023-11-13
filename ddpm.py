import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
#TODO: add utility class for DDPM

class DDPM:
    
    def __init__(self, beta_start = 0.0001, beta_end = 0.02, T = 3):
        
        self.T = T # Timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
     
    def beta_scheduler(self):    
        # Betas 
        return torch.linspace(self.beta_start, self.beta_end, self.T, dtype = torch.float64) 
    
    def alphas(self):
    # Alphas
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = np.append(1., self.alpha_bar[:-1])
        assert self.alpha_bar_prev.shape == (self.T,)
        # return 
    

    def noise_function(x0, t):
        # torch.rand_like(something) = Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.     
        noise = torch.randn_like(x0)
        return x0 + noise*t    
        

    def sample_timestep(self, batchsize):
        # Sampling t from a uniform distribution, what batchsize are we using?
        sampled_steps = torch.randint(1, self.T, (batchsize, )) 
        return sampled_steps      
    
    
    
    
       
        


 
        

