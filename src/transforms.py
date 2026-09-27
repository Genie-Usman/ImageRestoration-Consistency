import random
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import torch

def apply_consistency_transform(
    x,
    y_hat=None,
    angle_range=(-12, 12),
    scale_range=(0.85, 1.0),
    ratio_range=(0.9, 1.1),
    gain_range=(0.85, 1.15),
):
    """
    Applies identical geometric and photometric transformations to paired batch tensors.

    Parameters:
        x (torch.Tensor): Input batch [B, C, H, W] (e.g. low-light input).
        y_hat (torch.Tensor, optional): Corresponding output batch [B, C, H, W] (e.g. model prediction).

    Returns:
        tuple(torch.Tensor, torch.Tensor) or torch.Tensor:
            If y_hat is provided, returns (x_transformed, y_hat_transformed), where both
            share the EXACT SAME spatial rotation and resized crop for geometric equivariance.
            If y_hat is None, returns x_transformed.
    """
    out_x = []
    out_y = [] if y_hat is not None else None
    B, C, H, W = x.shape

    for i in range(B):
        img_x = x[i]

        # 1. Photometric jitter (exposure variation on input low-light image)
        gain = random.uniform(*gain_range)
        img_x = torch.clamp(img_x * gain, 0.0, 1.0)

        # 2. Geometric rotation
        angle = random.uniform(*angle_range)
        img_x = TF.rotate(img_x, angle, interpolation=InterpolationMode.BILINEAR)

        # 3. Geometric resized crop
        i0, j0, h0, w0 = T.RandomResizedCrop.get_params(
            img_x, scale=scale_range, ratio=ratio_range
        )
        img_x = TF.resized_crop(
            img_x, i0, j0, h0, w0, size=(H, W), interpolation=InterpolationMode.BILINEAR
        )
        img_x = torch.clamp(img_x, 0.0, 1.0)
        out_x.append(img_x)

        # 4. Apply IDENTICAL geometric transformation to y_hat for equivariance
        if y_hat is not None:
            img_y = y_hat[i]
            img_y = TF.rotate(img_y, angle, interpolation=InterpolationMode.BILINEAR)
            img_y = TF.resized_crop(
                img_y, i0, j0, h0, w0, size=(H, W), interpolation=InterpolationMode.BILINEAR
            )
            img_y = torch.clamp(img_y, 0.0, 1.0)
            out_y.append(img_y)

    x_T = torch.stack(out_x, dim=0)
    if y_hat is not None:
        y_T = torch.stack(out_y, dim=0)
        return x_T, y_T
    return x_T


def apply_transform_batch(x):
    """Backwards-compatible wrapper."""
    return apply_consistency_transform(x, y_hat=None)

