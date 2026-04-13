from neuralop.data.datasets import NavierStokesDataset


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