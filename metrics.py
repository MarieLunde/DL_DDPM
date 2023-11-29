import torch
#from torchmetrics.functional import inception_score as tm_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def fid_score(real_img_samples, generated_img_samples):
    #print("devices")
    
    real_img_samples = real_img_samples.to('cpu')
    #print(real_img_samples.get_device())
    
    generated_img_samples= generated_img_samples.to('cpu')
    #print(generated_img_samples.get_device())
    
    real_img_samples = real_img_samples.type(torch.uint8)
    generated_img_samples = generated_img_samples.type(torch.uint8)
    
    fid = FrechetInceptionDistance(feature=2048)

    fid.update(real_img_samples, real=True)
    fid.update(generated_img_samples, real=False)
    fidscore = fid.compute()

    return fidscore

def inception_score(generated_img_samples):
    
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