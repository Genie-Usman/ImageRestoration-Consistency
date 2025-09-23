import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class PairedImageDataset(Dataset):
    def __init__(self, low_dir, high_dir, patch_size=256, augment=True):
        self.low_paths = sorted(glob(os.path.join(low_dir, '*')))
        self.high_paths = sorted(glob(os.path.join(high_dir, '*')))
        assert len(self.low_paths) == len(self.high_paths), "mismatch count"
        self.patch = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert('RGB')
        high = Image.open(self.high_paths[idx]).convert('RGB')
        low = TF.to_tensor(low)
        high = TF.to_tensor(high)
        i, j, h, w = T.RandomCrop.get_params(low, output_size=(self.patch, self.patch))
        low = TF.crop(low, i, j, h, w)
        high = TF.crop(high, i, j, h, w)
        if self.augment and random.random() < 0.5:
            low = TF.hflip(low); high = TF.hflip(high)
        if self.augment and random.random() < 0.5:
            low = TF.vflip(low); high = TF.vflip(high)
        return {'low': low, 'high': high}
