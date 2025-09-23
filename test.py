import torch
import torchvision
import numpy as np
from PIL import Image
from lpips import LPIPS
from tqdm import tqdm

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
