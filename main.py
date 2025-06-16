"""Entry‑point script. Example: `python main.py --epochs 5`."""
import argparse
from torch.utils.data import DataLoader
from datasets import DummyImageDataset
from trainer import TextVAETrainer
from config import VAEConfig

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = VAEConfig(batch_size=args.batch)
    train_ds = DummyImageDataset(1000)
    val_ds = DummyImageDataset(200)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    trainer = TextVAETrainer(cfg)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
