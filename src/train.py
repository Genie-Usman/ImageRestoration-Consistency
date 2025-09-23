import os
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
from skimage.metrics import structural_similarity as ssim
import lpips
import csv

def compute_metrics(y_hat, y, lpips_fn, device):
    """Compute PSNR, SSIM, LPIPS for a batch."""
    y_hat_np = y_hat.detach().cpu().numpy().transpose(0, 2, 3, 1)
    y_np = y.detach().cpu().numpy().transpose(0, 2, 3, 1)

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    for i in range(y.shape[0]):
        psnr_scores.append(psnr(y_hat[i], y[i]).item())
        ssim_scores.append(
            ssim(y_np[i], y_hat_np[i], channel_axis=-1, data_range=1.0)
        )
        lp = lpips_fn(y_hat[i].unsqueeze(0).to(device), y[i].unsqueeze(0).to(device))
        lpips_scores.append(lp.item())

    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)


def main():
    # Paths
    train_low = 'data/train/low'
    train_high = 'data/train/high'
    val_low = 'data/val/low'
    val_high = 'data/val/high'
    os.makedirs('experiments/checkpoints', exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Datasets
    ds = PairedImageDataset(train_low, train_high, patch_size=256, augment=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    val_ds = PairedImageDataset(val_low, val_high, patch_size=256, augment=False)
    vdl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=1)

    # Model + loss
    model = UNetTiny().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loss_fn = PerceptualConsistencyLoss(device=device)

    # LPIPS metric
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Training config
    epochs = 100
    patience = 10
    best_lpips = float("inf")
    no_imp_epochs = 0

    # CSV logging
    log_file = 'experiments/metrics_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "ValPSNR", "ValSSIM", "ValLPIPS"])

    # Training loop
    for ep in range(epochs):
        model.train()
        running_loss = []
        loop = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        for batch in loop:
            x = batch['low'].to(device)
            y = batch['high'].to(device)
            opt.zero_grad()

            # forward
            y_hat = model(x)
            x_T = apply_transform_batch(x)
            y_hat_T_pred = model(x_T)
            y_hat_T_gt = apply_transform_batch(y_hat.detach())

            # loss
            loss = loss_fn(y_hat, y, y_hat_T_pred, y_hat_T_gt)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())
            loop.set_postfix(loss=np.mean(running_loss))

        # Validation
        model.eval()
        val_psnr, val_ssim, val_lpips = [], [], []
        with torch.no_grad():
            for vb in vdl:
                x = vb['low'].to(device)
                y = vb['high'].to(device)
                y_hat = model(x)
                p, s, l = compute_metrics(y_hat, y, lpips_fn, device)
                val_psnr.append(p); val_ssim.append(s); val_lpips.append(l)

        mean_psnr = np.mean(val_psnr)
        mean_ssim = np.mean(val_ssim)
        mean_lpips = np.mean(val_lpips)

        print(f"Epoch {ep+1}: "
              f"TrainLoss={np.mean(running_loss):.4f}, "
              f"PSNR={mean_psnr:.3f}, SSIM={mean_ssim:.3f}, LPIPS={mean_lpips:.3f}")

        # save metrics
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep+1, np.mean(running_loss), mean_psnr, mean_ssim, mean_lpips])

        # scheduler
        scheduler.step(mean_lpips)

        # save best checkpoint
        if mean_lpips < best_lpips:
            best_lpips = mean_lpips
            no_imp_epochs = 0
            torch.save(model.state_dict(), 'experiments/checkpoints/best.pth')
            print(f"✅ Saved new best model at epoch {ep+1} (LPIPS={best_lpips:.4f})")
        else:
            no_imp_epochs += 1

        # early stopping
        if no_imp_epochs >= patience:
            print(f"⛔ Early stopping at epoch {ep+1} (no improvement for {patience} epochs)")
            break

    # final model
    torch.save(model.state_dict(), 'experiments/checkpoints/final.pth')
    print("🎉 Training finished. Final model saved.")


if __name__ == '__main__':
    main()