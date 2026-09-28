import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from model import UNetTiny, GatedUNet
from utils import psnr
from skimage.metrics import structural_similarity as ssim
import lpips

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0)

def draw_header(image, title, subtext=""):
    """Adds a clean header banner over the image section."""
    draw = ImageDraw.Draw(image)
    banner_height = 42 if subtext else 28
    draw.rectangle([0, 0, image.width, banner_height], fill=(20, 22, 28))
    draw.text((12, 6), title, fill=(255, 255, 255))
    if subtext:
        draw.text((12, 22), subtext, fill=(80, 225, 130))
    return image

def create_4way_strip(low_img, base_img, ours_img, high_img, 
                      base_metrics=None, ours_metrics=None):
    """Combines [Input | Baseline | Ours | Ground Truth] into a single 4-panel strip."""
    W, H = low_img.size
    
    f_low = draw_header(low_img.copy(), "(a) Input (Low-Light)")
    
    b_sub = f"PSNR: {base_metrics['psnr']:.2f}dB | SSIM: {base_metrics['ssim']:.3f} | LPIPS: {base_metrics['lpips']:.3f}" if base_metrics else ""
    f_base = draw_header(base_img.copy(), "(b) Baseline UNetTiny", b_sub)
    
    o_sub = f"PSNR: {ours_metrics['psnr']:.2f}dB | SSIM: {ours_metrics['ssim']:.3f} | LPIPS: {ours_metrics['lpips']:.3f}" if ours_metrics else ""
    f_ours = draw_header(ours_img.copy(), "(c) Ours (Proposed)", o_sub)
    
    f_high = draw_header(high_img.copy(), "(d) Ground Truth (Normal-Light)")

    canvas = Image.new("RGB", (W * 4 + 30, H), color=(15, 16, 20))
    canvas.paste(f_low, (0, 0))
    canvas.paste(f_base, (W + 10, 0))
    canvas.paste(f_ours, (W * 2 + 20, 0))
    canvas.paste(f_high, (W * 3 + 30, 0))
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Generate 4-Panel Research Figures for Paper")
    parser.add_argument("--ours_checkpoint", type=str, default="experiments/checkpoints_unettiny_scratch/best.pth", help="Proposed model checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str, default="experiments/checkpoints/best.pth", help="Baseline model checkpoint")
    parser.add_argument("--input_folder", type=str, default="data/val/low", help="Low-light images directory")
    parser.add_argument("--gt_folder", type=str, default="data/val/high", help="Ground truth directory")
    parser.add_argument("--output_folder", type=str, default="output_research_figures", help="Output directory")
    parser.add_argument("--max_images", type=int, default=15, help="Number of comparison figures to generate")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load Proposed Model
    model_ours = UNetTiny().to(device)
    if os.path.exists(args.ours_checkpoint):
        state = torch.load(args.ours_checkpoint, map_location=device, weights_only=False)
        model_ours.load_state_dict(state, strict=False)
        print(f"Loaded Proposed model: {args.ours_checkpoint}")
    model_ours.eval()

    # Load Baseline Model
    model_base = UNetTiny().to(device)
    if os.path.exists(args.baseline_checkpoint):
        state = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        model_base.load_state_dict(state, strict=False)
        print(f"Loaded Baseline model: {args.baseline_checkpoint}")
    model_base.eval()

    # LPIPS evaluator
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    valid_exts = (".png", ".jpg", ".jpeg")
    filenames = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith(valid_exts)])[:args.max_images]

    print(f"\nGenerating 4-panel publication strips into '{args.output_folder}'...")

    for fname in filenames:
        low_path = os.path.join(args.input_folder, fname)
        high_path = os.path.join(args.gt_folder, fname)
        out_path = os.path.join(args.output_folder, f"figure_{os.path.splitext(fname)[0]}.png")

        x = load_image(low_path).to(device)
        y_gt = load_image(high_path).to(device)

        with torch.no_grad():
            y_base = model_base(x)
            y_ours = model_ours(x)

            # Compute Baseline metrics
            b_psnr = psnr(y_base, y_gt).item()
            b_np = y_base.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = y_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
            b_ssim = ssim(gt_np, b_np, channel_axis=-1, data_range=1.0)
            b_lpips = lpips_fn(y_base, y_gt, normalize=True).item()

            # Compute Proposed metrics
            o_psnr = psnr(y_ours, y_gt).item()
            o_np = y_ours.squeeze(0).permute(1, 2, 0).cpu().numpy()
            o_ssim = ssim(gt_np, o_np, channel_axis=-1, data_range=1.0)
            o_lpips = lpips_fn(y_ours, y_gt, normalize=True).item()

        base_metrics = {"psnr": b_psnr, "ssim": b_ssim, "lpips": b_lpips}
        ours_metrics = {"psnr": o_psnr, "ssim": o_ssim, "lpips": o_lpips}

        low_pil = Image.open(low_path).convert("RGB")
        base_pil = to_pil(y_base.squeeze(0).clamp(0, 1).cpu())
        ours_pil = to_pil(y_ours.squeeze(0).clamp(0, 1).cpu())
        high_pil = Image.open(high_path).convert("RGB")

        strip = create_4way_strip(low_pil, base_pil, ours_pil, high_pil, base_metrics, ours_metrics)
        strip.save(out_path)
        print(f"  Figure saved: {out_path} | Baseline LPIPS: {b_lpips:.3f} -> Ours LPIPS: {o_lpips:.3f}")

    print(f"\nAll publication figures successfully created in: {args.output_folder}")

if __name__ == "__main__":
    main()
