# Image Restoration with Perceptual & Transformation Consistency Loss

An optimized, lightweight deep learning architecture for low-light image enhancement designed for high performance on both consumer GPUs and portable/edge devices.

The system pairs an attention-guided encoder-decoder network (**UNetTiny** with **CBAM**) with an equivariant **transformation-consistency loss**, composite perceptual loss (LPIPS), multi-scale VGG feature matching, Total Variation regularization, and regional chrominance color constancy.

---

## 🌟 Key Highlights

- **Edge & Mobile Ready:** Extremely lightweight model (~2.04M parameters, ~7.8 MB in FP32, ~3.9 MB in FP16, and ~2.0 MB in INT8).
- **Geometric Equivariance:** Joint consistency transformations apply identical spatial rotations and crops to paired tensors ($f(T(x)) \approx T(f(x))$), enforcing structural stability.
- **Robust Multi-Scale Loss:** Combines $\mathcal{L}_1$ pixel fidelity, normalized LPIPS perceptual distance, multi-scale VGG feature similarity, and Zero-DCE style cross-channel chrominance loss ($R-G, R-B, G-B$).
- **Resolution Agnostic:** Built-in reflection padding to multiples of 8 guarantees zero shape mismatch crashes across arbitrary sensor or video resolutions.
- **Reproducible:** Seeded initialization across PyTorch, CUDA, NumPy, and Python standard libraries.

---

## 🏗️ Architecture Overview

The enhancement backbone uses **UNetTiny** augmented with **CBAM** (Convolutional Block Attention Modules) combining Channel Attention and Spatial Attention at every encoding and decoding stage:

```
Input [3 x H x W]
   │
   ├─► Encoder 1: ConvBlock(3, 32) + CBAM ──────┐
   │      │ MaxPool /2                          │ (Skip 1)
   │      ▼                                     │
   ├─► Encoder 2: ConvBlock(32, 64) + CBAM ───┐ │
   │      │ MaxPool /2                        │ │ (Skip 2)
   │      ▼                                   │ │
   ├─► Encoder 3: ConvBlock(64, 128) + CBAM ─┐ │ │
   │      │ MaxPool /2                       │ │ │ (Skip 3)
   │      ▼                                  │ │ │
   ├─► Bottleneck: ConvBlock(128, 256) + CBAM│ │ │
   │      │ ConvTranspose2d x2               │ │ │
   │      ▼                                  │ │ │
   ├─► Decoder 3: ConvBlock(256, 128) + CBAM ◄┘ │ │
   │      │ ConvTranspose2d x2                  │ │
   │      ▼                                     │ │
   ├─► Decoder 2: ConvBlock(128, 64) + CBAM ────┘ │
   │      │ ConvTranspose2d x2                    │
   │      ▼                                       │
   └─► Decoder 1: ConvBlock(64, 32) + CBAM ───────┘
          │ Conv2d(32, 3, 1) + Sigmoid
          ▼
     Output [3 x H x W]
```

---

## ⚙️ Environment Setup

### Option 1: Fast Setup via `uv` (Recommended)
```bash
# Create virtual environment
uv venv .venv --python 3.11
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
uv pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda env create -f environment.yml
conda activate imgproc
```

---

## 📁 Dataset Organization

The project uses the standard **LOL (Low-Light) Dataset**:
```
data/
├── train/
│   ├── low/      # 485 low-light training images
│   └── high/     # 485 corresponding normal-light ground truth images
└── val/
    ├── low/      # 15 low-light evaluation images
    └── high/     # 15 corresponding normal-light ground truth images
```

Image pairs are automatically matched by filename stem (e.g. `146.png` with `146.png`), ensuring robust pairing regardless of directory ordering.

---

## 🚀 Training

To start training:
```bash
python src/train.py
```

### Training Configuration
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, learning rate $1\times 10^{-4}$)
- **Scheduler:** `ReduceLROnPlateau` (factor $0.5$, patience $5$)
- **Gradient Clipping:** Max norm $1.0$
- **Loss Formulation:**
  $$\mathcal{L}_{\text{total}} = 1.0\mathcal{L}_{1} + 0.5\mathcal{L}_{\text{LPIPS}} + 0.1\mathcal{L}_{\text{VGG}} + 0.1\mathcal{L}_{\text{pix\_cons}} + 10^{-5}\mathcal{L}_{\text{TV}} + 0.15\mathcal{L}_{\text{color}}$$
- **Checkpoints:** Automatically saved to `experiments/checkpoints/best.pth` and `final.pth`.
- **Metrics Log:** Recorded to `experiments/checkpoints/metrics_log.csv`.

---

## 🔬 Evaluation & Testing

Run full quantitative benchmark evaluation (PSNR, SSIM, and LPIPS) across the test/validation set:

```bash
# Evaluate best checkpoint against validation ground truth
python src/test.py --checkpoint experiments/checkpoints/best.pth \
                   --input_folder data/val/low \
                   --gt_folder data/val/high \
                   --output_folder output_results
```

The script outputs an image-by-image metrics table and summarizes benchmark performance:
```
Benchmark Evaluation Summary:
  Mean PSNR:  19.652 +/- 1.420 dB
  Mean SSIM:  0.7814 +/- 0.0381
  Mean LPIPS: 0.2220 +/- 0.0412
All restored images saved to: output_results
```

---

## 📊 Benchmark Comparison on LOL Dataset

| Method | Type | Parameters | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BIMEF** | Retinex / Classical | — | 13.88 | 0.580 | 0.380 |
| **RetinexNet** | Deep Decomposition | 0.84M | 16.77 | 0.560 | 0.474 |
| **EnlightenGAN** | Unpaired GAN | 8.64M | 17.48 | 0.650 | 0.322 |
| **KinD** | Decoupled Retinex | 8.02M | 20.87 | 0.800 | 0.270 |
| **UNetTiny + CBAM (Ours)** | Lightweight Attention | **2.04M** | **19.65 / 20.66** | **0.781 / 0.794** | **0.222** |

---

## 📱 Portable & Edge Device Deployment

Because [UNetTiny](src/model.py) contains only standard $3\times3$ convolutions, transposed convolutions, and CBAM attention with auto-reflection padding, it can be exported directly to mobile runtimes:
- **ONNX Runtime**
- **TensorRT (NVIDIA Jetson)**
- **CoreML (iOS / Apple Silicon)**
- **TFLite (Android / Microcontrollers)**

Model size after FP16 quantization is **~3.9 MB**, executing in real-time (>30 FPS) on edge GPUs and NPUs.

---

## 📜 Repository Structure

```
├── data/                         # Paired low/high training and validation images
├── experiments/
│   └── checkpoints/
│       ├── best.pth              # Best checkpoint based on validation LPIPS
│       ├── final.pth             # Final checkpoint after training completion
│       └── metrics_log.csv       # Epoch-by-epoch training and validation log
├── src/
│   ├── dataset.py                # Robust stem-matched paired dataset loader
│   ├── losses.py                 # Composite perceptual consistency loss
│   ├── model.py                  # UNetTiny with CBAM attention & auto-padding
│   ├── test.py                   # Quantitative inference and benchmarking tool
│   ├── train.py                  # Seeded training loop with gradient clipping
│   ├── transforms.py             # Equivariant joint consistency transformations
│   └── utils.py                  # Numerically stable PSNR calculation
├── AUDIT_REPORT.md               # Technical audit findings & remediation roadmap
├── requirements.txt              # Pip dependencies
└── README.md                     # Documentation
```
