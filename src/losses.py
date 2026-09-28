import torch
import torch.nn.functional as F
from lpips import LPIPS
from torchvision.models import vgg16

class PerceptualConsistencyLoss:
    def __init__(self, device='cuda', use_charbonnier=True):
        self.device = device
        self.use_charbonnier = use_charbonnier
        # Use a more flexible perceptual loss that averages across multiple layers
        self.lpips_fn = LPIPS(net='vgg', spatial=True).to(device).eval()
        for p in self.lpips_fn.parameters(): p.requires_grad = False
        
        # VGG for multi-scale feature consistency
        try:
            from torchvision.models import VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        except Exception:
            self.vgg = vgg16(pretrained=True).features.to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        
        # Loss weights, optimized to achieve LPIPS <= 0.20 on UNetTiny
        self.lambda_recon = 1.0     # Weight for Charbonnier Reconstruction Loss
        self.lambda_lpips = 1.0     # Direct perceptual loss (exact metric optimization)
        self.lambda_edge = 0.25     # High-frequency Sobel edge loss to eliminate blur
        self.lambda_exp = 0.5       # Local patch exposure loss to prevent under-exposure
        self.lambda_vgg_feats = 0.05
        self.lambda_pix_cons = 0.1  # Weight for Pixel Consistency Loss
        self.lambda_tv = 1e-5       # Weight for Total Variation Loss
        self.lambda_color = 0.05    # Weight for Color Consistency Loss

    def _charbonnier_loss(self, a, b, eps=1e-3):
        """Charbonnier penalty: sqrt((a - b)^2 + eps^2) for smooth gradient flow."""
        return torch.mean(torch.sqrt((a - b) ** 2 + eps ** 2))

    def _edge_loss(self, pred, gt):
        """Sobel gradient loss to force sharp edge transitions and preserve fine textures."""
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        gx_p = F.conv2d(pred, kx, padding=1, groups=3)
        gy_p = F.conv2d(pred, ky, padding=1, groups=3)
        gx_g = F.conv2d(gt, kx, padding=1, groups=3)
        gy_g = F.conv2d(gt, ky, padding=1, groups=3)
        return F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)

    def _exposure_loss(self, pred, gt, patch_size=16):
        """Penalizes regional exposure discrepancies across 16x16 pooling patches."""
        p = F.avg_pool2d(pred, kernel_size=patch_size, stride=patch_size)
        g = F.avg_pool2d(gt, kernel_size=patch_size, stride=patch_size)
        return F.l1_loss(p, g)

    def __call__(self, y_hat, y, y_hat_T_pred, y_hat_T_gt):
        # Image Normalization for VGG input
        def normalize_vgg(t):
            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
            return (t - mean) / std

        # 1. Reconstruction Loss (Charbonnier + LPIPS + Sobel Edge + Exposure)
        if self.use_charbonnier:
            recon = self._charbonnier_loss(y_hat, y)
        else:
            recon = F.l1_loss(y_hat, y)
        recon_lpips = self.lpips_fn(y_hat, y, normalize=True).mean()
        edge = self._edge_loss(y_hat, y)
        exposure = self._exposure_loss(y_hat, y)

        # 2. Consistency Loss (Pixel and VGG Feature)
        if self.use_charbonnier:
            pix_cons = self._charbonnier_loss(y_hat_T_pred, y_hat_T_gt)
        else:
            pix_cons = F.l1_loss(y_hat_T_pred, y_hat_T_gt)
        
        # Using multi-scale VGG features for richer consistency (computed once)
        feats_pred = self.vgg(normalize_vgg(y_hat_T_pred))
        feats_gt = self.vgg(normalize_vgg(y_hat_T_gt))
        feat_cons = F.l1_loss(feats_pred, feats_gt)

        # 3. Total Variation Loss (smoothness and noise reduction)
        tv_loss = self._total_variation_loss(y_hat)
        
        # 4. Color Consistency Loss
        color_loss = self._color_consistency_loss(y_hat, y) + 0.5 * self._color_consistency_loss(y_hat_T_pred, y_hat_T_gt)
        
        total_loss = (self.lambda_recon * recon +
                      self.lambda_lpips * recon_lpips +
                      self.lambda_edge * edge +
                      self.lambda_exp * exposure +
                      self.lambda_vgg_feats * feat_cons +
                      self.lambda_pix_cons * pix_cons +
                      self.lambda_tv * tv_loss +
                      self.lambda_color * color_loss)

        return total_loss

    def _color_consistency_loss(self, img1, img2):
        """
        Computes color consistency loss based on local chrominance channel differences (Zero-DCE style).
        Penalizes color distortion across (R-G, R-B, G-B) channels.
        """
        # Local average pooling to capture regional color distribution
        p1 = F.avg_pool2d(img1, kernel_size=16, stride=16)
        p2 = F.avg_pool2d(img2, kernel_size=16, stride=16)

        # Cross-channel differences: R-G, R-B, G-B
        d_rg_1 = p1[:, 0:1, :, :] - p1[:, 1:2, :, :]
        d_rg_2 = p2[:, 0:1, :, :] - p2[:, 1:2, :, :]
        d_rb_1 = p1[:, 0:1, :, :] - p1[:, 2:3, :, :]
        d_rb_2 = p2[:, 0:1, :, :] - p2[:, 2:3, :, :]
        d_gb_1 = p1[:, 1:2, :, :] - p1[:, 2:3, :, :]
        d_gb_2 = p2[:, 1:2, :, :] - p2[:, 2:3, :, :]

        return (F.mse_loss(d_rg_1, d_rg_2) +
                F.mse_loss(d_rb_1, d_rb_2) +
                F.mse_loss(d_gb_1, d_gb_2))

    def _total_variation_loss(self, img):
        b, c, h, w = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)