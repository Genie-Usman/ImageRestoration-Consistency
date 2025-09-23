import random
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import torch

def apply_transform_batch(x):
    out = []
    B = x.shape[0]
    for i in range(B):
        img = x[i]

        # random brightness scaling
        gain = random.uniform(0.85, 1.15)
        img2 = img * gain

        # random rotation
        angle = random.uniform(-12, 12)
        img2 = TF.rotate(img2, angle, interpolation=InterpolationMode.BILINEAR)

        # random resized crop
        i0, j0, h0, w0 = T.RandomResizedCrop.get_params(
            img2, scale=(0.85, 1.0), ratio=(0.9, 1.1)
        )
        img2 = TF.resized_crop(
            img2, i0, j0, h0, w0, size=(img.shape[1], img.shape[2]), interpolation=InterpolationMode.BILINEAR
        )

        # clamp to valid pixel range
        img2 = torch.clamp(img2, 0.0, 1.0)

        out.append(img2)

    return torch.stack(out, dim=0)
