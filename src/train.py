import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import PairedImageDataset
from model import UNetTiny
from losses import PerceptualConsistencyLoss
from transforms import apply_transform_batch
from utils import psnr

def main():
    train_low = 'data/train/low'
    train_high = 'data/train/high'
    val_low = 'data/val/low'
    val_high = 'data/val/high'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = PairedImageDataset(train_low, train_high, patch_size=256, augment=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    val_ds = PairedImageDataset(val_low, val_high, patch_size=256, augment=False)
    vdl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=1)

    model = UNetTiny().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PerceptualConsistencyLoss(device=device)

    epochs = 1
    for ep in range(epochs):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        for batch in loop:
            x = batch['low'].to(device)
            y = batch['high'].to(device)
            opt.zero_grad()
            y_hat = model(x)
            x_T = apply_transform_batch(x)
            y_hat_T_pred = model(x_T)
            y_hat_T_gt = apply_transform_batch(y_hat.detach())
            loss = loss_fn(y_hat, y, y_hat_T_pred, y_hat_T_gt)
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item())

        # validation logging
        model.eval()
        with torch.no_grad():
            psnr_list = []
            for vb in vdl:
                x = vb['low'].to(device); y = vb['high'].to(device)
                y_hat = model(x)
                psnr_list.append(psnr(y_hat, y).item())
            print(f"Epoch {ep+1} val PSNR: {np.mean(psnr_list):.3f}")

    torch.save(model.state_dict(), 'experiments/checkpoints/model_final.pth')
    print("Model saved.")

if __name__ == '__main__':
    main()
