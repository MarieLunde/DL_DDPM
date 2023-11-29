import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
#from torchmetrics.functional import inception_score as tm_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_activation_statistics(images, model, device):
    model.eval()
    activations = model(images)
    activations = activations.view(activations.size(0), -1)
    mean = torch.mean(activations, dim=0)
    cov = torch_cov(activations, rowvar=False)
    return mean, cov

def torch_cov(m, rowvar=False):
    # Compute covariance matrix
    # m: [batch_size, num_features]
    if rowvar:
        m = m.t()
    # Calculate covariance matrix
    fact = 1.0 / (m.size(0) - 1)
    m -= torch.mean(m, dim=0, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

def fid_score(real_img_samples, generated_img_samples, device):
    
    fid = FrechetInceptionDistance(feature=2048)
    
    # Load the pretrained InceptionV3 model
    model = inception_v3(pretrained=True, aux_logits=False)
    model.to(device)

    real_img_samples = real_img_samples.to(device)
    generated_img_samples = generated_img_samples.to(device)
    
    mean1, cov1 = calculate_activation_statistics(real_img_samples, model, device)
    mean2, cov2 = calculate_activation_statistics(generated_img_samples, model, device)

    imgs_dist1 = {'mean': mean1, 'cov': cov1}
    imgs_dist2 = {'mean': mean2, 'cov': cov2}
    
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    fidscore = fid.compute()

    return fidscore