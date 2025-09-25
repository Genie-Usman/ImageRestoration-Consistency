import torch
import torch.nn.functional as F
from lpips import LPIPS
from torchvision.models import vgg16

class PerceptualConsistencyLoss:
    def __init__(self, device='cuda'):
        self.device = device
        # Use a more flexible perceptual loss that averages across multiple layers
        self.lpips_fn = LPIPS(net='vgg', spatial=True).to(device).eval()
        
        # VGG for multi-scale feature consistency
        self.vgg = vgg16(pretrained=True).features.to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        
        # Loss weights, optimized based on low-light enhancement practices
        self.lambda_l1 = 1.0
        self.lambda_lpips = 0.5
        self.lambda_vgg_feats = 0.1
        self.lambda_tv = 1e-5  # Weight for Total Variation Loss
        self.lambda_color = 0.1 # Weight for Color Consistency Loss

    def __call__(self, y_hat, y, y_hat_T_pred, y_hat_T_gt):
        # Image Normalization for VGG input
        def normalize_vgg(t):
            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
            return (t - mean) / std

        # 1. Reconstruction Loss (L1 + LPIPS)
        recon_l1 = F.l1_loss(y_hat, y)
        recon_lpips = self.lpips_fn(y_hat, y).mean()

        # 2. Consistency Loss (Pixel and VGG Feature)
        pix_cons = F.l1_loss(y_hat_T_pred, y_hat_T_gt)
        
        # Using multi-scale VGG features for richer consistency
        feats_pred = self.vgg(normalize_vgg(y_hat_T_pred))
        feats_gt = self.vgg(normalize_vgg(y_hat_T_gt))
        feat_cons = F.l1_loss(feats_pred, feats_gt)  # Use L1 for stability

        # 3. Total Variation Loss (for smoothness and noise reduction)
        tv_loss = self._total_variation_loss(y_hat)
        
        # 4. Color Consistency Loss (using VGG features for color channels)
        # This can be a separate VGG feature loss on the transformed images
        color_loss = F.l1_loss(self.vgg(normalize_vgg(y_hat_T_pred))[:, :64], self.vgg(normalize_vgg(y_hat_T_gt))[:, :64])
        
        total_loss = (self.lambda_l1 * recon_l1 +
                      self.lambda_lpips * recon_lpips +
                      self.lambda_vgg_feats * feat_cons +
                      0.1 * pix_cons + # Adjust this weight
                      self.lambda_tv * tv_loss +
                      self.lambda_color * color_loss)

        return total_loss

    def _total_variation_loss(self, img):
        b, c, h, w = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)