from neuralop.data.datasets import NavierStokesDataset
import matplotlib.pyplot as plt
import torch

dataset = NavierStokesDataset(
    root_dir='comp395final_data',
    n_train=1000,
    n_tests=[200],
    batch_size=16,
    test_batch_sizes=[16],
    train_resolution=128,
    test_resolutions=[128],
    download=True   # <-- fetches from Zenodo automatically
)
train_loader = dataset.train_loader
test_loaders = dataset.test_loaders


# Grab a batch from the test loader
test_loader = dataset.test_loaders[128]
batch = next(iter(test_loader))

x = batch['x']  # input vorticity fields
y = batch['y']  # target vorticity fields

# Plot a few samples
n_samples = 4
fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))

for i in range(n_samples):
    # Input (first time step channel)
    im0 = axes[i, 0].imshow(x[i, 0].cpu(), cmap='RdBu_r')
    axes[i, 0].set_title(f'Sample {i} — Input (vorticity in)')
    axes[i, 0].axis('off')
    plt.colorbar(im0, ax=axes[i, 0])

    # Target
    im1 = axes[i, 1].imshow(y[i, 0].cpu(), cmap='RdBu_r')
    axes[i, 1].set_title(f'Sample {i} — Target (vorticity out)')
    axes[i, 1].axis('off')
    plt.colorbar(im1, ax=axes[i, 1])

plt.tight_layout()
plt.savefig('ns_samples.png', dpi=150)
plt.show()