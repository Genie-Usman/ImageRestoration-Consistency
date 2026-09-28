import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import PairedImageDataset
from model import UNetTiny, GatedUNet
from losses import PerceptualConsistencyLoss
from transforms import apply_consistency_transform, apply_transform_batch
from utils import psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import csv
import argparse

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(y_hat, y, lpips_fn, device):
    """Compute PSNR, SSIM, LPIPS for a batch."""
    y_hat_np = y_hat.detach().cpu().numpy().transpose(0, 2, 3, 1)
    y_np = y.detach().cpu().numpy().transpose(0, 2, 3, 1)

    psnr_scores, ssim_scores = [], []
    for i in range(y.shape[0]):
        psnr_scores.append(psnr(y_hat[i], y[i]).item())
        ssim_scores.append(
            ssim(y_np[i], y_hat_np[i], channel_axis=-1, data_range=1.0)
        )

    with torch.no_grad():
        lpips_scores = lpips_fn(y_hat, y, normalize=True).flatten().detach().cpu().tolist()

    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)


def main():
    parser = argparse.ArgumentParser(description="Train Lightweight Restoration Model with Consistency Loss")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "gated"], help="Model architecture")
    parser.add_argument("--init_weights", type=str, default="", help="Pretrained checkpoint (leave empty to train strictly from scratch)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch size = 8)")
    parser.add_argument("--patch_size", type=int, default=256, help="Crop patch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate for training from scratch")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--residual", action="store_true", default=False, help="Enable global residual learning")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints_unettiny_scratch", help="Directory to save checkpoints")
    args = parser.parse_args()

    set_seed(42)

    # Paths
    train_low = 'data/train/low'
    train_high = 'data/train/high'
    val_low = 'data/val/low'
    val_high = 'data/val/high'
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print(f"Architecture: {args.arch} | Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum}) | Patch: {args.patch_size}x{args.patch_size}")

    # Datasets: Train on patches, validate on full-resolution 400x600 images
    ds = PairedImageDataset(train_low, train_high, patch_size=args.patch_size, augment=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    val_ds = PairedImageDataset(val_low, val_high, patch_size=None, augment=False)
    vdl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Training config
    epochs = args.epochs
    patience = args.patience
    best_lpips = float("inf")
    no_imp_epochs = 0

    # Model + loss
    if args.arch == "gated":
        model = GatedUNet(global_residual=args.residual).to(device)
    else:
        model = UNetTiny(global_residual=args.residual).to(device)

    # Load baseline weights only if explicitly provided, else train strictly from scratch
    if args.init_weights and os.path.exists(args.init_weights):
        try:
            state = torch.load(args.init_weights, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[INIT] Loaded pretrained weights from '{args.init_weights}'.")
    else:
        print("[INIT] Training model strictly FROM SCRATCH with random initialization (Reproducible Research Protocol).")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    loss_fn = PerceptualConsistencyLoss(device=device, use_charbonnier=True)

    # Automatic Mixed Precision (AMP) for RTX 2050 Tensor Cores
    use_amp = (device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # LPIPS metric (call with normalize=True for [0, 1] inputs)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # CSV logging
    log_file = os.path.join(args.save_dir, 'metrics_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "ValPSNR", "ValSSIM", "ValLPIPS", "LR"])

    # Training loop
    for ep in range(epochs):
        model.train()
        running_loss = []
        loop = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        opt.zero_grad()
        for b_idx, batch in enumerate(loop):
            x = batch['low'].to(device)
            y = batch['high'].to(device)

            # Forward + consistency under mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    y_hat = model(x)
                    x_T, y_hat_T_gt = apply_consistency_transform(x, y_hat.detach())
                    y_hat_T_pred = model(x_T)
                    raw_loss = loss_fn(y_hat, y, y_hat_T_pred, y_hat_T_gt)
                    loss = raw_loss / args.grad_accum
            else:
                y_hat = model(x)
                x_T, y_hat_T_gt = apply_consistency_transform(x, y_hat.detach())
                y_hat_T_pred = model(x_T)
                raw_loss = loss_fn(y_hat, y, y_hat_T_pred, y_hat_T_gt)
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()
            if (b_idx + 1) % args.grad_accum == 0 or (b_idx + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            running_loss.append(raw_loss.item())
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
        current_lr = opt.param_groups[0]['lr']

        print(f"Epoch {ep+1}: "
              f"TrainLoss={np.mean(running_loss):.4f}, "
              f"PSNR={mean_psnr:.3f}, SSIM={mean_ssim:.3f}, LPIPS={mean_lpips:.3f}, LR={current_lr:.6f}")

        # save metrics
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep+1, np.mean(running_loss), mean_psnr, mean_ssim, mean_lpips, current_lr])

        # Step cosine scheduler
        scheduler.step()

        # save best checkpoint
        if mean_lpips < best_lpips:
            best_lpips = mean_lpips
            no_imp_epochs = 0
            best_path = os.path.join(args.save_dir, 'best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Saved new best model at epoch {ep+1} (LPIPS={best_lpips:.4f}) to {best_path}")
        else:
            no_imp_epochs += 1

        # early stopping
        if no_imp_epochs >= patience:
            print(f"[STOP] Early stopping at epoch {ep+1} (no improvement for {patience} epochs)")
            break

    # final model
    final_path = os.path.join(args.save_dir, 'final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"[DONE] Training finished. Final model saved to {final_path}")


if __name__ == '__main__':
    main()