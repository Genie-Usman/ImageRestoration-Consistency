import torch
import torch.nn.functional as F
from lpips import LPIPS
from torchvision.models import vgg16

class PerceptualConsistencyLoss:
    def __init__(self, device='cuda'):
        self.lpips_fn = LPIPS(net='vgg').to(device).eval()
        self.vgg = vgg16(pretrained=True).features.to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        self.device = device

    def __call__(self, y_hat, y, y_hat_T_pred, y_hat_T_gt):
        # reconstruction loss
        recon_l1 = F.l1_loss(y_hat, y)
        recon_lpips = self.lpips_fn(y_hat, y).mean()
        # pixel consistency
        pix_cons = F.l1_loss(y_hat_T_pred, y_hat_T_gt)
        # feature consistency
        def prep_for_vgg(t):
            mean = torch.tensor([0.485,0.456,0.406], device=t.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=t.device).view(1,3,1,1)
            return (t - mean) / std
        feats_pred = self.vgg(prep_for_vgg(y_hat_T_pred))
        feats_gt   = self.vgg(prep_for_vgg(y_hat_T_gt))
        feat_cons = F.mse_loss(feats_pred, feats_gt)
        return recon_l1 + 0.01*recon_lpips + 0.1*pix_cons + 0.05*feat_cons
