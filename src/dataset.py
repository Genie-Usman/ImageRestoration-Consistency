import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

class PairedImageDataset(Dataset):
    """
    Dataset for paired low-light and normal-light images.
    Matches image pairs robustly by filename stem.
    Supports random cropping & flipping during training,
    and deterministic center cropping or full-image evaluation during validation.
    """
    def __init__(self, low_dir, high_dir, patch_size=256, augment=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.patch = patch_size
        self.augment = augment

        # Find all valid images
        low_files = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(low_dir, '*'))
            if p.lower().endswith(VALID_EXTENSIONS)
        }
        high_files = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(high_dir, '*'))
            if p.lower().endswith(VALID_EXTENSIONS)
        }

        # Pair matching by file stem
        common_keys = sorted(set(low_files.keys()) & set(high_files.keys()))
        unpaired_low = set(low_files.keys()) - set(high_files.keys())
        unpaired_high = set(high_files.keys()) - set(low_files.keys())

        if unpaired_low:
            print(f"[Warning] Found {len(unpaired_low)} unpaired low-light images: {unpaired_low}")
        if unpaired_high:
            print(f"[Warning] Found {len(unpaired_high)} unpaired normal-light images: {unpaired_high}")

        assert len(common_keys) > 0, f"No matching image pairs found between '{low_dir}' and '{high_dir}'"

        self.pairs = [(low_files[k], high_files[k]) for k in common_keys]
        self.low_paths = [p[0] for p in self.pairs]
        self.high_paths = [p[1] for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low = Image.open(low_path).convert('RGB')
        high = Image.open(high_path).convert('RGB')

        low = TF.to_tensor(low)
        high = TF.to_tensor(high)

        # Spatial cropping
        if self.patch is not None and self.patch > 0:
            c, h, w = low.shape
            if h >= self.patch and w >= self.patch:
                if self.augment:
                    # Random crop during training
                    i, j, th, tw = T.RandomCrop.get_params(low, output_size=(self.patch, self.patch))
                    low = TF.crop(low, i, j, th, tw)
                    high = TF.crop(high, i, j, th, tw)
                else:
                    # Deterministic center crop during validation if patch specified
                    low = TF.center_crop(low, (self.patch, self.patch))
                    high = TF.center_crop(high, (self.patch, self.patch))

        # Training augmentations
        if self.augment:
            if random.random() < 0.5:
                low = TF.hflip(low)
                high = TF.hflip(high)
            if random.random() < 0.5:
                low = TF.vflip(low)
                high = TF.vflip(high)

        return {'low': low, 'high': high}
