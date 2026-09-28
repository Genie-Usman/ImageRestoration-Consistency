import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0)

def get_fonts():
    """Load crisp TrueType fonts with robust OS fallback."""
    font_candidates = [
        ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"),
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    ]
    for bold_path, regular_path in font_candidates:
        if os.path.exists(bold_path) and os.path.exists(regular_path):
            try:
                title_font = ImageFont.truetype(bold_path, 22)
                sub_font = ImageFont.truetype(regular_path, 17)
                return title_font, sub_font
            except Exception:
                continue
    d = ImageFont.load_default()
    return d, d

def create_panel(img, title, metrics_text="", is_ours=False, title_font=None, sub_font=None):
    """Creates a clean image panel with a dedicated non-overlapping header."""
    W, H = img.size
    header_h = 66
    panel = Image.new("RGB", (W, H + header_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(panel)

    # Accent top border
    if is_ours:
        draw.rectangle([0, 0, W, 4], fill=(70, 220, 130))
    else:
        draw.rectangle([0, 0, W, 2], fill=(45, 52, 64))

    # Title
    draw.text((16, 10), title, font=title_font, fill=(255, 255, 255))

    # Subtext / Metrics
    if metrics_text:
        text_color = (95, 235, 145) if is_ours else (195, 205, 220)
        draw.text((16, 38), metrics_text, font=sub_font, fill=text_color)

    # Paste full uncropped image below header
    panel.paste(img, (0, header_h))
    return panel

ZOOM_BOXES = {
    "1": (480, 50, 565, 135),    # Cat line drawing
    "146": (190, 190, 290, 290),  # Text and textures on book
    "179": (240, 180, 340, 280),  # Lamp and edge details
    "493": (260, 150, 370, 260),  # Plush toy face
    "780": (180, 180, 320, 320),  # Center floor reflection
}

def add_zoom_inset(img_pil, box, zoom_size=160):
    """Draws a red bounding box and displays a 2.5x zoomed-in crop in the corner."""
    if box is None:
        return img_pil
    W, H = img_pil.size
    crop = img_pil.crop(box).resize((zoom_size, zoom_size), Image.BILINEAR)
    res = img_pil.copy()
    draw = ImageDraw.Draw(res)
    # Bounding box on original region
    draw.rectangle(box, outline=(255, 45, 45), width=3)
    # Inset window in bottom-right corner
    x0, y0 = W - zoom_size - 12, H - zoom_size - 12
    res.paste(crop, (x0, y0))
    draw.rectangle((x0, y0, x0 + zoom_size, y0 + zoom_size), outline=(255, 45, 45), width=3)
    return res

def create_multi_strip(panels_data, title_font=None, sub_font=None, zoom_box=None):
    """
    Combines N panels into a single publication strip.
    panels_data: list of dicts with keys:
      - 'img': PIL.Image
      - 'title': str
      - 'metrics': dict with psnr, ssim, lpips (optional)
      - 'is_ours': bool
    """
    rendered_panels = []
    for p in panels_data:
        img = p['img']
        if zoom_box is not None:
            img = add_zoom_inset(img, zoom_box)
        
        m_text = ""
        if p.get('metrics'):
            m = p['metrics']
            m_text = f"PSNR: {m['psnr']:.2f} dB   SSIM: {m['ssim']:.3f}   LPIPS: {m['lpips']:.3f}"
        elif p.get('subtext'):
            m_text = p['subtext']
            
        panel = create_panel(img, p['title'], m_text, p.get('is_ours', False), title_font, sub_font)
        rendered_panels.append(panel)

    panel_w, panel_h = rendered_panels[0].size
    n_panels = len(rendered_panels)
    spacing = 10
    total_w = panel_w * n_panels + spacing * (n_panels - 1)
    canvas = Image.new("RGB", (total_w, panel_h), color=(12, 14, 18))
    for i, pan in enumerate(rendered_panels):
        canvas.paste(pan, (i * (panel_w + spacing), 0))
    return canvas

def eval_metrics_pair(pred_img_pil, gt_img_pil, lpips_fn, device):
    """Computes exact PSNR, SSIM, and LPIPS between two PIL images."""
    gt_np = np.array(gt_img_pil)
    pred_np = np.array(pred_img_pil)
    if pred_np.shape != gt_np.shape:
        pred_img_pil = pred_img_pil.resize((gt_np.shape[1], gt_np.shape[0]), Image.BICUBIC)
        pred_np = np.array(pred_img_pil)

    p = compute_psnr(gt_np, pred_np, data_range=255)
    s = compute_ssim(gt_np, pred_np, channel_axis=2, data_range=255)

    t_gt = to_tensor(gt_img_pil).unsqueeze(0).to(device)
    t_pred = to_tensor(pred_img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        l = lpips_fn(t_pred * 2 - 1, t_gt * 2 - 1).item()

    return {"psnr": p, "ssim": s, "lpips": l}

def main():
    parser = argparse.ArgumentParser(description="Generate Research & Competition Figures for Paper")
    parser.add_argument("--mode", type=str, default="sota", choices=["ablation", "sota", "sota_transformer"],
                        help="Comparison mode: 'ablation' (4-way internal) or 'sota' (6-way competition)")
    parser.add_argument("--ours_dir", type=str, default="output_results_scratch_best", help="Proposed model output directory")
    parser.add_argument("--input_folder", type=str, default="data/val/low", help="Low-light images directory")
    parser.add_argument("--gt_folder", type=str, default="data/val/high", help="Ground truth directory")
    parser.add_argument("--output_folder", type=str, default="output_sota_figures", help="Output directory")
    parser.add_argument("--max_images", type=int, default=15, help="Number of comparison figures to generate")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running visualizer in '{args.mode}' mode on device: {device}")

    # LPIPS evaluator
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # Load high-resolution publication fonts
    title_font, sub_font = get_fonts()

    # Model directories
    benchmarks_root = "experiments/benchmarks"
    ruas_dir = os.path.join(benchmarks_root, "LOL_v1/LOL_v1_compare_origin_img/RUAS_LOL_v1")
    engan_dir = os.path.join(benchmarks_root, "EnlightenGAN_LOL_v1")
    kind_dir = os.path.join(benchmarks_root, "LOL_v1/LOL_v1_compare_origin_img/KinD_LOL_v1")
    restormer_dir = os.path.join(benchmarks_root, "LOL_v1/LOL_v1_compare_origin_img/Restormer_Lolv1/best_psnr_22")
    base_dir = "output_results"

    valid_exts = (".png", ".jpg", ".jpeg")
    filenames = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith(valid_exts)])[:args.max_images]

    print(f"\nGenerating multi-panel publication strips into '{args.output_folder}'...")

    for fname in filenames:
        low_path = os.path.join(args.input_folder, fname)
        high_path = os.path.join(args.gt_folder, fname)
        out_path = os.path.join(args.output_folder, f"figure_{os.path.splitext(fname)[0]}.png")

        low_pil = Image.open(low_path).convert("RGB")
        high_pil = Image.open(high_path).convert("RGB")
        stem = os.path.splitext(fname)[0]
        zoom_box = ZOOM_BOXES.get(stem, None)

        if args.mode == "ablation":
            # 4-Way Strip: [Input | Baseline | Ours | Ground Truth]
            base_pil = Image.open(os.path.join(base_dir, fname)).convert("RGB")
            ours_pil = Image.open(os.path.join(args.ours_dir, fname)).convert("RGB")

            base_metrics = eval_metrics_pair(base_pil, high_pil, lpips_fn, device)
            ours_metrics = eval_metrics_pair(ours_pil, high_pil, lpips_fn, device)

            panels = [
                {"img": low_pil, "title": "(a) Input (Low-Light)", "subtext": "Raw Degraded Input", "is_ours": False},
                {"img": base_pil, "title": "(b) Baseline UNetTiny", "metrics": base_metrics, "is_ours": False},
                {"img": ours_pil, "title": "(c) Ours (Proposed)", "metrics": ours_metrics, "is_ours": True},
                {"img": high_pil, "title": "(d) Ground Truth", "subtext": "Reference Normal-Light", "is_ours": False},
            ]
        elif args.mode == "sota_transformer":
            # 6-Way Strip: [Input | EnlightenGAN | KinD | Restormer | Ours | Ground Truth]
            engan_pil = Image.open(os.path.join(engan_dir, fname)).convert("RGB")
            kind_pil = Image.open(os.path.join(kind_dir, fname)).convert("RGB")
            restormer_pil = Image.open(os.path.join(restormer_dir, fname)).convert("RGB")
            ours_pil = Image.open(os.path.join(args.ours_dir, fname)).convert("RGB")

            engan_m = eval_metrics_pair(engan_pil, high_pil, lpips_fn, device)
            kind_m = eval_metrics_pair(kind_pil, high_pil, lpips_fn, device)
            restormer_m = eval_metrics_pair(restormer_pil, high_pil, lpips_fn, device)
            ours_m = eval_metrics_pair(ours_pil, high_pil, lpips_fn, device)

            panels = [
                {"img": low_pil, "title": "(a) Input (Low-Light)", "subtext": "Raw Degraded Input", "is_ours": False},
                {"img": engan_pil, "title": "(b) EnlightenGAN", "metrics": engan_m, "is_ours": False},
                {"img": kind_pil, "title": "(c) KinD", "metrics": kind_m, "is_ours": False},
                {"img": restormer_pil, "title": "(d) Restormer (26.1M)", "metrics": restormer_m, "is_ours": False},
                {"img": ours_pil, "title": "(e) Ours (2.04M)", "metrics": ours_m, "is_ours": True},
                {"img": high_pil, "title": "(f) Ground Truth", "subtext": "Reference Normal-Light", "is_ours": False},
            ]
        else:
            # Standard 6-Way SOTA Strip: [Input | RUAS | EnlightenGAN | KinD | Ours | Ground Truth]
            ruas_pil = Image.open(os.path.join(ruas_dir, fname)).convert("RGB")
            engan_pil = Image.open(os.path.join(engan_dir, fname)).convert("RGB")
            kind_pil = Image.open(os.path.join(kind_dir, fname)).convert("RGB")
            ours_pil = Image.open(os.path.join(args.ours_dir, fname)).convert("RGB")

            ruas_m = eval_metrics_pair(ruas_pil, high_pil, lpips_fn, device)
            engan_m = eval_metrics_pair(engan_pil, high_pil, lpips_fn, device)
            kind_m = eval_metrics_pair(kind_pil, high_pil, lpips_fn, device)
            ours_m = eval_metrics_pair(ours_pil, high_pil, lpips_fn, device)

            panels = [
                {"img": low_pil, "title": "(a) Input (Low-Light)", "subtext": "Raw Degraded Input", "is_ours": False},
                {"img": ruas_pil, "title": "(b) RUAS (CVPR '21)", "metrics": ruas_m, "is_ours": False},
                {"img": engan_pil, "title": "(c) EnlightenGAN (TIP '21)", "metrics": engan_m, "is_ours": False},
                {"img": kind_pil, "title": "(d) KinD (MM '19)", "metrics": kind_m, "is_ours": False},
                {"img": ours_pil, "title": "(e) Ours (UNetTiny+CBAM)", "metrics": ours_m, "is_ours": True},
                {"img": high_pil, "title": "(f) Ground Truth", "subtext": "Reference Normal-Light", "is_ours": False},
            ]

        strip = create_multi_strip(panels, title_font=title_font, sub_font=sub_font, zoom_box=zoom_box)
        strip.save(out_path)
        print(f"  [OK] Saved: {out_path}")

    print(f"\nAll publication figures successfully created in: {args.output_folder}")

if __name__ == "__main__":
    main()
