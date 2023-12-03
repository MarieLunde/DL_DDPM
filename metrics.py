import torch
from torchmetrics.image.inception import InceptionScore

def preprocess_fid_score(images):
    if images.shape[1] == 1: # MNIST
        images = torch.cat((images, images, images), axis=1)
    images = images.to('cpu') 
    images = images.type(torch.uint8)
    return images

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