import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CBAM Block
# -----------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size,
                                      padding=kernel_size // 2,
                                      bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # ----- Channel Attention -----
        avg_pool = torch.mean(x, dim=(2, 3))   # [B, C]
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)  # [B, C]

        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_att = self.sigmoid_channel(channel_att).view(b, c, 1, 1)
        x = x * channel_att

        # ----- Spatial Attention -----
        avg_out = torch.mean(x, dim=1, keepdim=True)       # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, H, W]
        spatial_att = self.sigmoid_spatial(
            self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_att

        return x


# -----------------------------
# Conv Block with CBAM
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)   # Apply attention
        return x


# -----------------------------
# Encoder-Decoder UNetTiny
# -----------------------------
class UNetTiny(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, global_residual=False):
        super().__init__()
        self.global_residual = global_residual

        # Encoder
        self.enc1 = ConvBlock(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Final output
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # Auto-pad to multiple of 8 if needed (prevents shape mismatch in pooling/decoder)
        H, W = x.shape[2], x.shape[3]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_in = x

        # Encoder
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.out_conv(d1)
        if self.global_residual:
            # Residual illumination enhancement: y = clamp(x + delta, 0, 1)
            out = torch.clamp(x_in + residual, 0.0, 1.0)
        else:
            # Direct prediction mode (compatible with previous checkpoints)
            out = torch.sigmoid(residual)

        # Unpad back to original spatial dimensions
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


# -----------------------------
# Lightweight Gated U-Net (SGU + NAF-style Blocks for High Perceptual Fidelity)
# -----------------------------
class SimpleGate(nn.Module):
    """Splits channel dimension into two halves and computes element-wise product."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """2D Spatial LayerNorm with learnable affine weights."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class GatedBlock(nn.Module):
    """
    Lightweight Gated Restoration Block (Gated DWConv + Simple Channel Attention + FFN).
    Provides high-frequency edge and texture preservation without heavy attention.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True)
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = y * self.conv5(y)
        return x + y * self.gamma


class GatedUNet(nn.Module):
    """
    Lightweight Gated U-Net (< 7 MB, ~1.6M params) for Edge Devices.
    Replaces MaxPool with learnable strided downsampling and ConvTranspose with PixelShuffle
    to eliminate checkerboard artifacts and retain rich textural details for LPIPS < 0.222.
    """
    def __init__(self, in_ch=3, out_ch=3, base_c=32, global_residual=False):
        super().__init__()
        self.global_residual = global_residual
        self.intro = nn.Conv2d(in_ch, base_c, 3, padding=1)

        # Encoder (learnable strided downsampling, no MaxPool)
        self.enc1 = GatedBlock(base_c)
        self.down1 = nn.Conv2d(base_c, base_c * 2, 2, stride=2)

        self.enc2 = GatedBlock(base_c * 2)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, 2, stride=2)

        self.enc3 = GatedBlock(base_c * 4)
        self.down3 = nn.Conv2d(base_c * 4, base_c * 8, 2, stride=2)

        # Bottleneck (Rich multi-scale context)
        self.mid1 = GatedBlock(base_c * 8)
        self.mid2 = GatedBlock(base_c * 8)

        # Decoder (PixelShuffle upsampling, zero checkerboard artifacts)
        self.up3 = nn.Sequential(nn.Conv2d(base_c * 8, base_c * 4 * 4, 1), nn.PixelShuffle(2))
        self.fuse3 = nn.Conv2d(base_c * 8, base_c * 4, 1)
        self.dec3 = GatedBlock(base_c * 4)

        self.up2 = nn.Sequential(nn.Conv2d(base_c * 4, base_c * 2 * 4, 1), nn.PixelShuffle(2))
        self.fuse2 = nn.Conv2d(base_c * 4, base_c * 2, 1)
        self.dec2 = GatedBlock(base_c * 2)

        self.up1 = nn.Sequential(nn.Conv2d(base_c * 2, base_c * 4, 1), nn.PixelShuffle(2))
        self.fuse1 = nn.Conv2d(base_c * 2, base_c, 1)
        self.dec1 = GatedBlock(base_c)

        self.outro = nn.Conv2d(base_c, out_ch, 3, padding=1)

    def forward(self, x):
        # Auto-reflection pad to multiple of 8
        H, W = x.shape[2], x.shape[3]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_in = x

        x1 = self.intro(x_in)
        e1 = self.enc1(x1)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        m = self.mid2(self.mid1(self.down3(e3)))

        d3 = self.dec3(self.fuse3(torch.cat([self.up3(m), e3], dim=1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], dim=1)))

        residual = self.outro(d1)
        if self.global_residual:
            out = torch.clamp(x_in + residual, 0.0, 1.0)
        else:
            out = torch.sigmoid(residual)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out