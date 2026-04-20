import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from dataset import NSForcingDataset
from models import FNO2d, UNet


def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=(-1, -2)) / torch.norm(target, dim=(-1, -2))).mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = NSForcingDataset('data/nsforcing_train_128.pt')
    test_ds = NSForcingDataset('data/nsforcing_test_128.pt')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    if args.model == 'fno':
        model = FNO2d(modes1=args.fno_modes, modes2=args.fno_modes,
                      width=args.fno_width, n_layers=args.fno_layers)
    else:
        model = UNet(base_channels=args.unet_channels)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = wandb.init(
        project='comp395-ns-benchmark',
        name=f"{args.model}_run",
        config=vars(args),
    )
    wandb.watch(model, log_freq=100)

    best_test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = rel_l2(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    test_loss += rel_l2(pred, y).item()
            test_loss /= len(test_loader)

            print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | test={test_loss:.4f}")
            wandb.log({'train/rel_l2': train_loss, 'test/rel_l2': test_loss, 'epoch': epoch})

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f"checkpoints/{args.model}_best.pt")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f}")
            wandb.log({'train/rel_l2': train_loss, 'epoch': epoch})

    run.finish()
    print(f"Best test rel-L2: {best_test_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fno', 'unet'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5)
    # FNO hyperparams
    parser.add_argument('--fno_modes', type=int, default=12)
    parser.add_argument('--fno_width', type=int, default=64)
    parser.add_argument('--fno_layers', type=int, default=4)
    # U-Net hyperparams
    parser.add_argument('--unet_channels', type=int, default=64)
    args = parser.parse_args()

    import os
    os.makedirs('checkpoints', exist_ok=True)

    train(args)
