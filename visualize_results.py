"""
Generate prediction comparison figures using the trained FNO and U-Net checkpoints.

Usage:
  uv run python visualize_results.py

Outputs (all saved to figures/):
  predictions.png     — 6 test samples: input | FNO pred | U-Net pred | ground truth
  error_maps.png      — absolute error |pred - truth| for FNO and U-Net, same 6 samples
  super_res.png       — FNO vs U-Net at 64×64, 128×128, 256×256 (one sample)
  error_histogram.png — per-sample rel-L2 distribution across full test set
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from dataset import NSForcingDataset
from models import FNO2d, UNet

os.makedirs('figures', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Load models ──────────────────────────────────────────────────────────────

fno = FNO2d(modes1=12, modes2=12, width=64, n_layers=4)
fno.load_state_dict(torch.load('checkpoints/fno_best.pt', map_location=DEVICE))
fno = fno.to(DEVICE).eval()

unet = UNet(base_channels=64)
unet.load_state_dict(torch.load('checkpoints/unet_best.pt', map_location=DEVICE))
unet = unet.to(DEVICE).eval()

# ── Load test data ────────────────────────────────────────────────────────────

ds = NSForcingDataset('data/nsforcing_test_128.pt')
loader = DataLoader(ds, batch_size=20, shuffle=False)

xs, ys, fnos, unets = [], [], [], []
with torch.no_grad():
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xs.append(xb.cpu()); ys.append(yb.cpu())
        fnos.append(fno(xb).cpu())
        unets.append(unet(xb).cpu())

x_all  = torch.cat(xs)
y_all  = torch.cat(ys)
fno_all  = torch.cat(fnos)
unet_all = torch.cat(unets)


def rel_l2_per_sample(pred, target):
    return (torch.norm(pred - target, dim=(-1, -2)) /
            torch.norm(target, dim=(-1, -2))).cpu().numpy()


fno_errs  = rel_l2_per_sample(fno_all,  y_all)
unet_errs = rel_l2_per_sample(unet_all, y_all)

# Pick 6 representative samples: sort by FNO error, pick evenly spaced
sorted_idx = np.argsort(fno_errs)
sample_idx = [sorted_idx[int(i * (len(sorted_idx) - 1) / 5)] for i in range(6)]

# ── Figure 1: predictions grid ────────────────────────────────────────────────

N = len(sample_idx)
fig, axes = plt.subplots(4, N, figsize=(3 * N, 13))
fig.suptitle('FNO vs U-Net Predictions on Navier–Stokes Test Set', fontsize=14, y=1.01)
row_labels = ['Input  ω(t)', 'FNO  pred', 'U-Net pred', 'Ground truth  ω(t+1)']

for col, idx in enumerate(sample_idx):
    x  = x_all[idx].cpu().numpy()
    y  = y_all[idx].cpu().numpy()
    fp = fno_all[idx].cpu().numpy()
    up = unet_all[idx].cpu().numpy()

    vmin = min(y.min(), fp.min(), up.min())
    vmax = max(y.max(), fp.max(), up.max())

    for row, (data, label) in enumerate(zip([x, fp, up, y], row_labels)):
        ax = axes[row, col]
        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        if row == 1:
            ax.set_title(f'FNO L2={fno_errs[idx]:.3f}', fontsize=9)
        if row == 2:
            ax.set_title(f'UNet L2={unet_errs[idx]:.3f}', fontsize=9)

fig.tight_layout()
fig.savefig('figures/predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/predictions.png")

# ── Figure 2: absolute error maps ─────────────────────────────────────────────

fig, axes = plt.subplots(2, N, figsize=(3 * N, 7))
fig.suptitle('Absolute Prediction Error  |pred − truth|', fontsize=13, y=1.01)

for col, idx in enumerate(sample_idx):
    y  = y_all[idx].cpu().numpy()
    fp = fno_all[idx].cpu().numpy()
    up = unet_all[idx].cpu().numpy()

    fno_err_map  = np.abs(fp - y)
    unet_err_map = np.abs(up - y)
    vmax = max(fno_err_map.max(), unet_err_map.max())

    for row, (emap, label) in enumerate(zip([fno_err_map, unet_err_map], ['FNO', 'U-Net'])):
        ax = axes[row, col]
        im = ax.imshow(emap, cmap='hot', vmin=0, vmax=vmax, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
        if row == 0:
            ax.set_title(f'Sample {idx}', fontsize=9)

fig.tight_layout()
fig.savefig('figures/error_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/error_maps.png")

# ── Figure 3: super-resolution comparison ────────────────────────────────────
# Pick the median-error sample

med_idx = sorted_idx[len(sorted_idx) // 2]
x_s = x_all[[med_idx]].to(DEVICE)
y_s = y_all[[med_idx]].to(DEVICE)

resolutions = [64, 128, 256]
fig, axes = plt.subplots(3, len(resolutions), figsize=(4 * len(resolutions), 10))
fig.suptitle('Zero-Shot Super-Resolution: FNO vs U-Net (trained only at 128×128)', fontsize=12)

for col, res in enumerate(resolutions):
    xr = F.interpolate(x_s.unsqueeze(1), size=(res, res), mode='bilinear',
                       align_corners=False).squeeze(1)
    yr = F.interpolate(y_s.unsqueeze(1), size=(res, res), mode='bilinear',
                       align_corners=False).squeeze(1)

    with torch.no_grad():
        fp = fno(xr)
        up = unet(xr)

    fno_l2  = (torch.norm(fp - yr, dim=(-1, -2)) / torch.norm(yr, dim=(-1, -2))).item()
    unet_l2 = (torch.norm(up - yr, dim=(-1, -2)) / torch.norm(yr, dim=(-1, -2))).item()

    vmin = yr[0].min().item()
    vmax = yr[0].max().item()

    for row, (data, label) in enumerate(zip(
        [yr[0].cpu().numpy(), fp[0].cpu().numpy(), up[0].cpu().numpy()],
        ['Ground truth', f'FNO  (L2={fno_l2:.3f})', f'U-Net (L2={unet_l2:.3f})']
    )):
        ax = axes[row, col]
        ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        if row == 0:
            tag = ' ← train res' if res == 128 else ''
            ax.set_title(f'{res}×{res}{tag}', fontsize=10)

fig.tight_layout()
fig.savefig('figures/super_res.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/super_res.png")

# ── Figure 4: per-sample error distribution ───────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(fno_errs.max(), unet_errs.max()) * 1.05, 40)
ax.hist(fno_errs,  bins=bins, alpha=0.7, label=f'FNO  (mean={fno_errs.mean():.4f})',  color='steelblue')
ax.hist(unet_errs, bins=bins, alpha=0.7, label=f'U-Net (mean={unet_errs.mean():.4f})', color='tomato')
ax.axvline(fno_errs.mean(),  color='steelblue', linestyle='--', linewidth=1.5)
ax.axvline(unet_errs.mean(), color='tomato',    linestyle='--', linewidth=1.5)
ax.set_xlabel('Per-sample relative L2 error', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Error Distribution on Full Test Set', fontsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('figures/error_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/error_histogram.png")

print("\nDone. All figures saved to figures/")
