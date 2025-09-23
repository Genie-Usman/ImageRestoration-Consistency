import torch

def psnr(a,b):
    mse = torch.mean((a-b)**2)
    return 10.0 * torch.log10(1.0 / mse)
