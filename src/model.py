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
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
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
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Encoder
        e1 = self.enc1(x)
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

        out = torch.sigmoid(self.out_conv(d1))

        # Unpad back to original spatial dimensions
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out