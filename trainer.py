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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from config import VAEConfig
from models import build_vae_from_config
from tqdm.auto import tqdm

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAETrainer:
    """
    Manages the training and validation loop for the TextVAE model.

    Handles device placement, optimization, learning rate scheduling,
    gradient accumulation, automatic mixed precision (AMP), and checkpointing.

    Args:
        cfg (VAE_config): The configuration object for the model and training.
        work_dir (str): Directory to save checkpoints and logs.
    """

    def __init__(self, cfg: VAEConfig, work_dir: str = "./runs"):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # --- Device Setup ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA is available. Training on GPU.")
        else:
            self.device = torch.device("cpu")
            logger.warning(
                "CUDA not available. Training on CPU. This will be very slow."
            )
            # AMP is only for CUDA
            if self.cfg.amp:
                logger.warning(
                    "AMP is enabled but CUDA is not available. Disabling AMP."
                )
                self.cfg.amp = False

        # --- Model, Optimizer, and Scaler ---
        self.model = build_vae_from_config(cfg).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        # GradScaler is used for stable gradients with float16 training (AMP).
        self.scaler = GradScaler(device=self.device.type, enabled=self.cfg.amp)

        self.global_step = 0
        logger.info(
            f"Trainer initialized. Checkpoints will be saved to {self.work_dir}"
        )

    def _run_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs a single forward pass and returns the model output."""
        batch = batch.to(self.device)
        # autocast enables automatic mixed precision for the forward pass.
        # Operations are run in float16 where safe, and float32 otherwise.
        with autocast(device_type=self.device.type, enabled=self.cfg.amp):
            model_output = self.model(batch)
            assert isinstance(
                model_output, dict
            ), f"Expected model output to be a dict, but got {type(model_output)}"

            out = cast(Dict[str, torch.Tensor], model_output)
            return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
    ):
        """
        The main training loop.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader, optional): DataLoader for the validation set.
            epochs (int): Number of epochs to train for.
        """
        logger.info(f"Starting training for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            self.model.train()
            # Use tqdm for a progress bar over the training loader.
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]")
            for i, batch in enumerate(pbar):
                # --- Gradient Accumulation Loop ---
                # The actual forward and backward pass.
                out = self._run_step(batch)
                loss = out["loss"]

                # Scale the loss for gradient accumulation.
                # This ensures the final accumulated gradient has the correct magnitude.
                scaled_loss = self.scaler.scale(
                    loss / self.cfg.gradient_accumulation_steps
                )
                scaled_loss.backward()

                # --- Optimizer Step ---
                # Step the optimizer only after accumulating enough gradients.
                if (i + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)  # More memory efficient
                    self.global_step += 1

                # Logging
                if self.global_step % self.cfg.log_every == 0:
                    metrics = {k: f"{v.item():.4f}" for k, v in out.items()}
                    pbar.set_postfix(metrics)

            # --- End of Epoch ---
            if val_loader:
                self.validate(val_loader, epoch)

            if epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

        logger.info("Training finished.")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int):
        """
        Runs a full validation loop, generating images and calculating
        FID and Perplexity metrics.
        """
        self.model.eval()
        logger.info(f"Running validation for epoch {epoch} with metrics...")

        all_ppl: List[float] = []

        # Create temporary directories to store real and generated images for FID
        with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
            real_path = Path(real_dir)
            fake_path = Path(fake_dir)
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Generating for Metrics]")
            img_idx = 0

            for batch in pbar:
                batch = batch.to(self.device)

                # --- Generate tokens and images ---
                # 1. Encode image to get generated tokens
                enc_out = self.model.encode(batch, sample=True)
                hard_tokens = enc_out["hard_tokens"]
                assert hard_tokens is not None

                # 2. Decode the generated tokens back to an image
                generated_images = self.model.decode(hard_tokens)

                # --- Calculate Perplexity ---
                # Get the LLM's predictions for the generated tokens
                with torch.no_grad():
                    lm_out = self.model.llm(input_ids=hard_tokens)
                    lm_logits = lm_out.last_hidden_state

                # Calculate cross-entropy loss against the LLM's own predictions
                # Shift logits to align with labels for next-token prediction
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = hard_tokens[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                perplexity = torch.exp(loss)
                all_ppl.append(perplexity.item())

                # --- Save images for FID calculation ---
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

            # --- Calculate FID Score ---
            logger.info("Calculating FID score... (this may take a moment)")
            # Set isc=False, fid=True, kid=False as we only need FID
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

        # --- Log final metrics ---
        avg_perplexity = sum(all_ppl) / len(all_ppl) if all_ppl else float("nan")
        log_str = (
            f"Validation Results Epoch {epoch} - "
            f"FID: {fid_score:.4f} | "
            f"Perplexity: {avg_perplexity:.4f}"
        )
        logger.info(log_str)

    def save_checkpoint(self, name: str):
        """
        Saves the model and optimizer state to a file.

        Args:
            name (str): The name for the checkpoint file.
        """
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
        """Loads a checkpoint to resume training."""
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
