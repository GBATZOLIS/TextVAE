# trainer.py

"""Training harness for TextVAE (supports AMP & gradient accumulation)."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, cast
import tempfile
import torch
from torch.amp import GradScaler, autocast
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from config import VAEConfig
from models import build_vae_from_config
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAETrainer:
    """Manages the training and validation loop for the TextVAE model."""

    def __init__(self, cfg: VAEConfig, work_dir: str = "./runs"):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA is available. Training on GPU.")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available. Training on CPU.")
            if self.cfg.amp:
                self.cfg.amp = False
                logger.warning("Disabling AMP since CUDA is not available.")

        self.model = build_vae_from_config(cfg).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        self.scaler = GradScaler(device=self.device.type, enabled=self.cfg.amp)
        self.global_step = 0
        logger.info(
            f"Trainer initialized. Checkpoints will be saved to {self.work_dir}"
        )

    def _run_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = batch.to(self.device)
        with autocast(device_type=self.device.type, enabled=self.cfg.amp):
            model_output = self.model(batch)
            out = cast(Dict[str, torch.Tensor], model_output)
            return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
    ):
        logger.info(f"Starting training for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]")
            for i, batch in enumerate(pbar):
                out = self._run_step(batch)
                loss = out["loss"]
                scaled_loss = self.scaler.scale(
                    loss / self.cfg.gradient_accumulation_steps
                )
                scaled_loss.backward()

                if (i + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1

                if self.global_step % self.cfg.log_every == 0:
                    metrics = {k: f"{v.item():.4f}" for k, v in out.items()}
                    pbar.set_postfix(metrics)

            if val_loader:
                self.validate(val_loader, epoch)
            if epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")
        logger.info("Training finished.")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        logger.info(f"Running validation for epoch {epoch} with metrics...")

        all_kl_loss: List[float] = []  # **MODIFIED**: Changed from perplexity to KL

        with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
            real_path = Path(real_dir)
            fake_path = Path(fake_dir)
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Generating for Metrics]")
            img_idx = 0

            for batch in pbar:
                batch = batch.to(self.device)

                enc_out = self.model.encode(batch, sample=True)
                hard_tokens = enc_out["hard_tokens"]
                assert hard_tokens is not None
                generated_images = self.model.decode(hard_tokens)

                # **MODIFIED**: Calculate validation KL divergence instead of perplexity
                kl_loss = self.model.compute_kl(enc_out["logits"], hard_tokens)
                all_kl_loss.append(kl_loss.item())

                for i in range(batch.size(0)):
                    vutils.save_image(
                        batch[i], real_path / f"{img_idx}.png", normalize=True
                    )
                    vutils.save_image(
                        generated_images[i],
                        fake_path / f"{img_idx}.png",
                        normalize=True,
                    )
                    img_idx += 1

            logger.info("Calculating FID score... (this may take a moment)")
            metrics_dict = calculate_metrics(
                str(fake_path),
                str(real_path),
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
            )
            fid_score = metrics_dict.get("frechet_inception_distance", float("nan"))

        # **MODIFIED**: Log the new metric
        avg_kl = sum(all_kl_loss) / len(all_kl_loss) if all_kl_loss else float("nan")
        log_str = (
            f"Validation Results Epoch {epoch} - "
            f"FID: {fid_score:.4f} | "
            f"Validation KL: {avg_kl:.4f}"
        )
        logger.info(log_str)

    def save_checkpoint(self, name: str):
        path = self.work_dir / name
        logger.info(f"Saving checkpoint to {path}")
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "cfg": self.cfg,
            "global_step": self.global_step,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            logger.error(f"Checkpoint not found at {path}")
            return
        logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        logger.info(f"Resumed training from global step {self.global_step}")
