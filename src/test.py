import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import UNetTiny
from utils import psnr
from skimage.metrics import structural_similarity as ssim

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0)  # shape: (1, C, H, W)

def save_image(tensor, path):
    img = to_pil(tensor.squeeze(0).clamp(0, 1).cpu())
    img.save(path)

def main():
    parser = argparse.ArgumentParser(description="Image Restoration Inference and Evaluation")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/best.pth", help="Path to checkpoint")
    parser.add_argument("--input_folder", type=str, default="data/val/low", help="Input low-light images directory")
    parser.add_argument("--gt_folder", type=str, default="data/val/high", help="Ground-truth normal-light images directory")
    parser.add_argument("--output_folder", type=str, default="output_results", help="Directory to save restored images")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = UNetTiny().to(device)
    if os.path.exists(args.checkpoint):
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        print(f"Checkpoint not found at '{args.checkpoint}'. Running with initialized weights.")

    model.eval()

    # Initialize LPIPS if ground truth evaluation is possible
    has_gt = os.path.isdir(args.gt_folder)
    lpips_fn = None
    if has_gt:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg", normalize=True).to(device)
        except Exception as e:
            print(f"LPIPS could not be loaded: {e}. Skipping LPIPS calculation.")

    psnr_list, ssim_list, lpips_list = [], [], []

    # Inference loop
    valid_exts = (".jpg", ".png", ".jpeg", ".bmp")
    filenames = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith(valid_exts)])

    print(f"\nProcessing {len(filenames)} images from '{args.input_folder}'...")
    print("-" * 75)
    header = f"{'Image':<15} | {'Saved Path':<25}"
    if has_gt:
        header += f" | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS':<8}"
    print(header)
    print("-" * 75)

    for fname in filenames:
        inp_path = os.path.join(args.input_folder, fname)
        out_path = os.path.join(args.output_folder, fname)

        x = load_image(inp_path).to(device)
        with torch.no_grad():
            y_hat = model(x)

        save_image(y_hat, out_path)

        # Ground truth evaluation
        row = f"{fname:<15} | {out_path:<25}"
        gt_path = os.path.join(args.gt_folder, fname) if has_gt else None
        if gt_path and os.path.exists(gt_path):
            y_gt = load_image(gt_path).to(device)

            # PSNR
            p_val = psnr(y_hat, y_gt).item()
            psnr_list.append(p_val)

            # SSIM
            y_hat_np = y_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
            y_gt_np = y_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
            s_val = ssim(y_gt_np, y_hat_np, channel_axis=-1, data_range=1.0)
            ssim_list.append(s_val)

            # LPIPS
            l_str = "N/A"
            if lpips_fn is not None:
                l_val = lpips_fn(y_hat, y_gt).item()
                lpips_list.append(l_val)
                l_str = f"{l_val:.4f}"

            row += f" | {p_val:<10.3f} | {s_val:<8.4f} | {l_str:<8}"

        print(row)

    print("-" * 75)
    if psnr_list:
        print("\nBenchmark Evaluation Summary:")
        print(f"  Mean PSNR:  {np.mean(psnr_list):.3f} +/- {np.std(psnr_list):.3f} dB")
        print(f"  Mean SSIM:  {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")
        if lpips_list:
            print(f"  Mean LPIPS: {np.mean(lpips_list):.4f} +/- {np.std(lpips_list):.4f}")
    print(f"\nAll restored images saved to: {args.output_folder}")

if __name__ == "__main__":
    main()