import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
#TODO: add utility class for DDPM

# I moved it inside the class
#def beta_scheduler(T):
#    beta_start =  0.0001
#    beta_end = 0.02
#    return torch.linspace(beta_start, beta_end, T, dtype = torch.float64)

class DDPM:
    
    def __init__(self, beta_start = 0.0001, beta_end = 0.02, T = 3):
        
        self.T = T # Timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Betas 
        self.betas = torch.linspace(beta_start, beta_end ,T, dtype=torch.float64) 
        # Alphas
        self.alphas = 1 - self.beta
        
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        alpha_bar_prev = np.append(1., alpha_bar[:-1])
        #assert alpha_bar_prev.shape == (T,)
    

    def noise_function(x0, t):
            
        noise = torch.randn_like(x0)
        return x0 + noise*t    
        # torch.rand_like(something) = Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. 

    def sample_timestep(self, batchsize):
        # Sampling t from a uniform distribution
        sampled_steps = torch.randint(1, self.T, (batchsize, )) # Are we sampling 1 or many???
        return sampled_steps      
    
    
    
    
       
        


 
        

