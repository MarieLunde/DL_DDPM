import torch
#from torchmetrics.functional import inception_score as tm_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance


def fid_score(real_img_samples, generated_img_samples):
    
    fid = FrechetInceptionDistance(feature=2048)

    fid.update(real_img_samples, real=True)
    fid.update(generated_img_samples, real=False)
    fidscore = fid.compute()

    return fidscore