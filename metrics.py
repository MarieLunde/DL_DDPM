import torch
#from torchmetrics.functional import inception_score as tm_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def fid_score(real_img_samples, generated_img_samples, dim=64): 
    if dim < generated_img_samples.shape[0]:
        print(f"there are less generated img samples ({generated_img_samples.shape[0]}) than features ({dim}). This might be last batch")
    if real_img_samples.shape[1] == 1: #MNIST
        real_img_samples = torch.cat((real_img_samples, real_img_samples, real_img_samples), axis=1)
        generated_img_samples = torch.cat((generated_img_samples, generated_img_samples, generated_img_samples), axis=1)
    real_img_samples = real_img_samples.to('cpu')    
    generated_img_samples= generated_img_samples.to('cpu')
    
    real_img_samples = real_img_samples.type(torch.uint8)
    generated_img_samples = generated_img_samples.type(torch.uint8)
    
    fid = FrechetInceptionDistance(feature=dim)

    fid.update(real_img_samples, real=True)
    fid.update(generated_img_samples, real=False)
    fidscore = fid.compute()

    return fidscore

def inception_score(generated_img_samples):
    if generated_img_samples.shape[1] == 1: #MNIST
        generated_img_samples = torch.cat((generated_img_samples, generated_img_samples, generated_img_samples), axis=1)
    generated_img_samples = generated_img_samples.type(torch.uint8)
    generated_img_samples= generated_img_samples.to('cpu')

    inception = InceptionScore()
    
    inception.update(generated_img_samples)
    inceptionscore = inception.compute()

    return inceptionscore



if __name__ == '__main__':
    x = torch.randn(3, 3, 32, 32)
    x2 = torch.randn(3, 3, 32, 32)

    fid = fid_score(x, x2)
    print(fid)