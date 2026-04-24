"""
Evaluate a trained model on the test set.

Standard evaluation (accuracy + inference speed):
  uv run python evaluate.py --model fno  --checkpoint checkpoints/fno_best.pt
  uv run python evaluate.py --model unet --checkpoint checkpoints/unet_best.pt

Zero-shot super-resolution (tests model at 64x64, 128x128, 256x256):
  uv run python evaluate.py --model fno  --checkpoint checkpoints/fno_best.pt  --super_res
  uv run python evaluate.py --model unet --checkpoint checkpoints/unet_best.pt --super_res
"""
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NSForcingDataset
from models import FNO2d, UNet


def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=(-1, -2)) / torch.norm(target, dim=(-1, -2))).mean()


def resize(t, size):
    # t: (batch, H, W) -> (batch, size, size)
    return F.interpolate(t.unsqueeze(1), size=(size, size), mode='bilinear',
                         align_corners=False).squeeze(1)


def build_model(args, device):
    if args.model == 'fno':
        model = FNO2d(modes1=args.fno_modes, modes2=args.fno_modes,
                      width=args.fno_width, n_layers=args.fno_layers)
    else:
        model = UNet(base_channels=args.unet_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    return model.to(device)


def eval_at_resolution(model, loader, res, device):
    """Run evaluation with inputs/targets resized to res x res."""
    total_loss = 0.0
    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if res != x.shape[-1]:
                x = resize(x, res)
                y = resize(y, res)

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

    return total_loss / total_samples, total_samples / total_time


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = NSForcingDataset('data/nsforcing_test_128.pt')
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = build_model(args, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model:      {args.model}")
    print(f"Parameters: {n_params:,}")
    print()

    if args.super_res:
        # Zero-shot super-resolution: test at resolutions the model never saw during training.
        # Inputs and ground-truth targets are bilinearly interpolated to each resolution.
        # FNO handles arbitrary resolution natively (Fourier ops are resolution-agnostic).
        # U-Net also runs at each resolution (as long as size is divisible by 16),
        # but is expected to degrade more since it has no resolution-invariant inductive bias.
        resolutions = [64, 128, 256]
        print(f"{'Resolution':<14} {'Rel-L2':>10} {'Throughput (samp/s)':>22}")
        print("-" * 48)
        for res in resolutions:
            loss, throughput = eval_at_resolution(model, loader, res, device)
            tag = " (train res)" if res == 128 else ""
            print(f"{res}x{res}{tag:<10} {loss:>10.4f} {throughput:>22.1f}")
    else:
        loss, throughput = eval_at_resolution(model, loader, 128, device)
        latency_ms = 1000.0 / throughput
        print(f"Test rel-L2: {loss:.4f}")
        print(f"Throughput:  {throughput:.1f} samples/sec")
        print(f"Latency:     {latency_ms:.3f} ms/sample")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fno', 'unet'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--super_res', action='store_true',
                        help='Test zero-shot super-resolution at 64x64, 128x128, 256x256')
    parser.add_argument('--fno_modes', type=int, default=12)
    parser.add_argument('--fno_width', type=int, default=64)
    parser.add_argument('--fno_layers', type=int, default=4)
    parser.add_argument('--unet_channels', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
