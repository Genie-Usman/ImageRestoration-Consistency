import torch

def psnr(a, b, max_val=1.0, eps=1e-10):
    mse = torch.mean((a - b) ** 2)
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))
