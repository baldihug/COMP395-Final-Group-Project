import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    U-Net baseline for 2D vorticity prediction.
    Input/output: (batch, 128, 128)
    """
    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.enc1 = _conv_block(1, c)
        self.enc2 = _conv_block(c, c * 2)
        self.enc3 = _conv_block(c * 2, c * 4)
        self.enc4 = _conv_block(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c * 2, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, 128, 128)
        x = x.unsqueeze(1)  # (batch, 1, 128, 128)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1).squeeze(1)  # (batch, 128, 128)
