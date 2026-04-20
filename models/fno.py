import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _mul(self, x, weights):
        # x: (batch, in_ch, modes1, modes2)
        # weights: (in_ch, out_ch, modes1, modes2)
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        bsz = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            bsz, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.bypass(x))


class FNO2d(nn.Module):
    """
    FNO for 2D vorticity prediction.
    Input: (batch, 128, 128) — adds grid coords to make (batch, 3, 128, 128)
    """
    def __init__(self, modes1=12, modes2=12, width=64, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(3, width, kernel_size=1)  # 1 vorticity + 2 grid
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.proj1 = nn.Conv2d(width, 128, kernel_size=1)
        self.proj2 = nn.Conv2d(128, 1, kernel_size=1)

    def _grid(self, bsz, size, device):
        g = torch.linspace(0, 1, size, device=device)
        gx, gy = torch.meshgrid(g, g, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)

    def forward(self, x):
        # x: (batch, 128, 128)
        bsz, h, w = x.shape
        x = x.unsqueeze(1)  # (batch, 1, 128, 128)
        grid = self._grid(bsz, h, x.device)
        x = torch.cat([x, grid], dim=1)  # (batch, 3, 128, 128)

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x.squeeze(1)  # (batch, 128, 128)
