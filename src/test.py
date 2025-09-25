import os
import torch
from PIL import Image
import torchvision.transforms as T
from model import UNetTiny

# same preprocessing as training
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0)  # shape: (1, C, H, W)

def save_image(tensor, path):
    img = to_pil(tensor.squeeze(0).clamp(0, 1).cpu())
    img.save(path)

def main():
    # paths
    checkpoint_path = "experiments/checkpoints/model_final.pth"
    input_folder = "data/val/low"       # run on validation low-light images
    output_folder = "output_results"
    os.makedirs(output_folder, exist_ok=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model
    model = UNetTiny().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # inference loop
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        inp_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        x = load_image(inp_path).to(device)
        with torch.no_grad():
            y_hat = model(x)

        save_image(y_hat, out_path)
        print(f"Saved restored image: {out_path}")

if _name_ == "_main_":
    main()