"""
Evaluate a trained model on the test set: relative L2 error and inference speed.
Usage:
  uv run python evaluate.py --model fno --checkpoint checkpoints/fno_best.pt
  uv run python evaluate.py --model unet --checkpoint checkpoints/unet_best.pt
"""
import argparse
import time
import torch
from torch.utils.data import DataLoader

from dataset import NSForcingDataset
from models import FNO2d, UNet


def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=(-1, -2)) / torch.norm(target, dim=(-1, -2))).mean()


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = NSForcingDataset('data/nsforcing_test_128.pt')
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    if args.model == 'fno':
        model = FNO2d(modes1=args.fno_modes, modes2=args.fno_modes,
                      width=args.fno_width, n_layers=args.fno_layers)
    else:
        model = UNet(base_channels=args.unet_channels)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_loss = 0.0
    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_loss += rel_l2(pred, y).item() * x.size(0)
            total_samples += x.size(0)
            total_time += (t1 - t0)

    mean_rel_l2 = total_loss / total_samples
    throughput = total_samples / total_time  # samples/sec

    print(f"Model:       {args.model}")
    print(f"Parameters:  {n_params:,}")
    print(f"Test rel-L2: {mean_rel_l2:.4f}")
    print(f"Throughput:  {throughput:.1f} samples/sec")
    print(f"Latency:     {1000 * total_time / total_samples:.3f} ms/sample")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fno', 'unet'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--fno_modes', type=int, default=12)
    parser.add_argument('--fno_width', type=int, default=64)
    parser.add_argument('--fno_layers', type=int, default=4)
    parser.add_argument('--unet_channels', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
