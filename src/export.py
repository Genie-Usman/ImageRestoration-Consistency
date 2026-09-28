import os
import argparse
import torch
from model import UNetTiny, GatedUNet

def export_onnx(checkpoint_path, output_path, height=400, width=600, opset=17, global_residual=False, arch="auto"):
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")

    state_dict = None
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)

    if arch == "auto":
        if state_dict is not None and any(k.startswith("intro") for k in state_dict.keys()):
            chosen_arch = "gated"
        else:
            chosen_arch = "unet"
    else:
        chosen_arch = arch

    print(f"Detected/Selected Architecture: '{chosen_arch}'")
    if chosen_arch == "gated":
        model = GatedUNet(global_residual=global_residual).to(device)
    else:
        model = UNetTiny(global_residual=global_residual).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print("Checkpoint weights loaded successfully.")
    else:
        print(f"[Warning] Checkpoint '{checkpoint_path}' not found. Exporting random initialized weights for testing.")

    model.eval()

    # Dummy input with specified resolution
    dummy_input = torch.randn(1, 3, height, width, requires_grad=False)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Exporting to ONNX format (opset {opset})...")

    # Dynamic axes support arbitrary image sizes on mobile/edge devices
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"SUCCESS: Exported ONNX model to: {output_path}")
    print(f"Model file size: {size_mb:.2f} MB")

    # Optional ONNX verification
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure verified successfully.")
    except ImportError:
        print("[Info] Install 'onnx' library ('pip install onnx') to run automated validation.")

    print("\n" + "=" * 60)
    print("Edge / Portable Deployment Quickstart:")
    print("=" * 60)
    print("1. Python (ONNX Runtime):")
    print("   import onnxruntime as ort")
    print("   session = ort.InferenceSession('model.onnx')")
    print("   output = session.run(['output'], {'input': img_np})")
    print("\n2. Android / iOS / C++:")
    print("   Deploy with Microsoft ONNX Runtime Mobile, Apple CoreML, or NCNN.")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Export UNetTiny to ONNX for Portable & Mobile Deployment")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/best.pth", help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, default="experiments/checkpoints/model.onnx", help="Target ONNX file path")
    parser.add_argument("--height", type=int, default=400, help="Default image height")
    parser.add_argument("--width", type=int, default=600, help="Default image width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--residual", action="store_true", help="Enable global residual learning mode")
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        height=args.height,
        width=args.width,
        opset=args.opset,
        global_residual=args.residual
    )

if __name__ == "__main__":
    main()
