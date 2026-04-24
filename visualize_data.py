import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

data_train = torch.load('data/nsforcing_train_128.pt', weights_only=False)
data_test  = torch.load('data/nsforcing_test_128.pt',  weights_only=False)

x_train = data_train['x']
y_train = data_train['y']

# --- Figure 1: input/target pairs for 5 samples ---
n = 5
fig, axes = plt.subplots(2, n, figsize=(15, 6))
fig.suptitle('Navier-Stokes Vorticity — Input vs Target (training samples)', fontsize=13)

for i in range(n):
    vmax = max(x_train[i].abs().max(), y_train[i].abs().max()).item()
    kw = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    im = axes[0, i].imshow(x_train[i].numpy(), **kw)
    axes[0, i].set_title(f'Input #{i}')
    axes[0, i].axis('off')
    plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    im = axes[1, i].imshow(y_train[i].numpy(), **kw)
    axes[1, i].set_title(f'Target #{i}')
    axes[1, i].axis('off')
    plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('figures/sample_pairs.png', dpi=120)
print("Saved figures/sample_pairs.png")

# --- Figure 2: difference (target - input) shows what the model must predict ---
fig, axes = plt.subplots(1, n, figsize=(15, 3))
fig.suptitle('Target − Input (what the model must learn)', fontsize=13)
for i in range(n):
    diff = (y_train[i] - x_train[i]).numpy()
    vmax = np.abs(diff).max()
    im = axes[i].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[i].set_title(f'Diff #{i}')
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('figures/sample_diffs.png', dpi=120)
print("Saved figures/sample_diffs.png")

# --- Figure 3: statistics ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Vorticity Value Distribution (training set)', fontsize=13)

axes[0].hist(x_train.flatten().numpy(), bins=100, color='steelblue', alpha=0.7, label='Input')
axes[0].hist(y_train.flatten().numpy(), bins=100, color='tomato', alpha=0.7, label='Target')
axes[0].set_xlabel('Vorticity')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].set_title('Histogram')

per_sample_std_x = x_train.std(dim=(-1,-2)).numpy()
per_sample_std_y = y_train.std(dim=(-1,-2)).numpy()
axes[1].scatter(per_sample_std_x[:500], per_sample_std_y[:500], alpha=0.3, s=5, color='purple')
axes[1].set_xlabel('Input std per sample')
axes[1].set_ylabel('Target std per sample')
axes[1].set_title('Per-sample std (first 500 samples)')

plt.tight_layout()
plt.savefig('figures/stats.png', dpi=120)
print("Saved figures/stats.png")
