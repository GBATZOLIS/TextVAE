"""Training harness for TextVAE (supports AMP & gradient accumulation)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from config import VAEConfig
from models import build_vae_from_config
from tqdm.auto import tqdm

class TextVAETrainer:
    def __init__(self, cfg: VAEConfig, work_dir: str = "./runs"):
        self.cfg = cfg
        self.model = build_vae_from_config(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
        self.scaler = GradScaler(enabled=cfg.amp)
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.llm_model_name)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def step(self, batch: torch.Tensor, train: bool = True) -> Dict[str, float]:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        if train:
            self.opt.zero_grad()
        with autocast(enabled=self.cfg.amp):
            out = self.model(batch)
        if train:
            self.scaler.scale(out["loss"].div(self.cfg.gradient_accumulation_steps)).backward()
            if (self.cfg.global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
        return {k: v.item() for k, v in out.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for i, batch in enumerate(pbar):
                metrics = self.step(batch, train=True)
                self.cfg.global_step += 1
                if i % self.cfg.log_every == 0:
                    pbar.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})
            if val_loader:
                self.validate(val_loader)
            if epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        device = next(self.model.parameters()).device
        losses = []
        for batch in val_loader:
            batch = batch.to(device)
            losses.append(self.model(batch)["loss"].item())
        print(f"Validation loss: {sum(losses)/len(losses):.4f}")

    def save_checkpoint(self, name: str):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(ckpt, self.work_dir / name)
