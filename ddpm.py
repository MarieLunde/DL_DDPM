# import numpy as np
import torch
from torch import nn, Tensor
# from torch.nn.functional import softplus
# from torch.distributions import Distribution
# import matplotlib.pyplot as plt
# from model import UNet



class DDPM(nn.Module):
  
    def __init__(self, beta_start = 0.0001, beta_end = 0.02, T = 1000, device='cpu'):
        super(DDPM, self).__init__()
        self.device = device
        self.T = T # Timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.beta_scheduler_linear()  
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat((torch.tensor([1.], device=device), self.alpha_bar[:-1]))

    
    def beta_scheduler_linear(self):     
        return torch.linspace(self.beta_start, self.beta_end, self.T, dtype = torch.float32, device=self.device) 

    def noising_function(self, x0, batchsize, model):
        t = torch.randint(1, self.T, (batchsize, ), device = self.device)
        noise = torch.randn_like(x0).to(self.device)
        sqrt_alpha_bar_x0 = (torch.sqrt(self.alpha_bar[t]).view(batchsize,1,1,1)).to(self.device)*x0
        sqrt_1_minus_alpha_bar = (torch.sqrt(1-self.alpha_bar[t]).view(batchsize,1,1,1)).to(self.device)
        noised_img = (sqrt_alpha_bar_x0 + sqrt_1_minus_alpha_bar*noise).to(self.device)
        noise_pred = model(noised_img, t)
        return noise_pred, noise

    def sampling_image(self, model, num_img, channels, img_shape):
        x = torch.randn((num_img, channels, img_shape, img_shape),device=self.device)
        model.eval()
        with torch.no_grad():
            for timestep in reversed(range(1, self.T)):
                t = (torch.ones(num_img) * timestep).long().to(self.device)
                if timestep > 1:
                    z = torch.randn_like(x)
                else:
                    z = 0
                var = (1 - self.alpha_bar[timestep - 1]) / (1 - self.alpha_bar[timestep]) * self.beta[timestep]
                pred_noise = model(x, t)
                model_mean = 1 / torch.sqrt(self.alpha[timestep]) * (x - ((1 - self.alpha[timestep]) / (torch.sqrt(1 - self.alpha_bar[timestep]))) * pred_noise)
                x = model_mean + torch.sqrt(var) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x0 = (x * 255).type(torch.uint8)
        return x0