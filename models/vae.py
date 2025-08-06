# models/vae.py
"""
The main Text VAE model.
This version uses a full VLM as the encoder and a frozen Diffusion model
as the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from config import AppConfig
from .vision_encoder import VLMEncoder


class TextVAE(nn.Module):
    """
    The main VAE-like model that orchestrates the VLM encoder and Diffusion decoder.
    """

    def __init__(self, cfg: AppConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- 1. The VLM Encoder ---
        self.encoder = VLMEncoder(cfg.models.vlm_id, device)

        # --- 2. The Frozen Diffusion Decoder Components ---
        self.vae = AutoencoderKL.from_pretrained(cfg.models.t2i_id, subfolder="vae").to(
            device
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.models.t2i_id, subfolder="unet"
        ).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(
            cfg.models.t2i_id, subfolder="scheduler"
        )

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # --- 3. Parameter-Efficient Fine-Tuning Setup ---
        self.encoder.model.vision_tower.requires_grad_(False)
        self.encoder.model.language_model.requires_grad_(False)

    def _kl_divergence_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates KL divergence against a uniform distribution."""
        log_q = F.log_softmax(logits, dim=-1)
        log_p = F.log_softmax(torch.ones_like(logits), dim=-1)
        return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)

    @torch.no_grad()
    def encode(self, image: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        **For Inference:** Encodes an image into its compressed "soft prompt" representation.
        """
        self.eval()
        soft_prompt_embeds, _ = self.encoder(image, temperature)
        return soft_prompt_embeds

    @torch.no_grad()
    def decode(
        self, soft_prompt_embeds: torch.Tensor, num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        **For Inference:** Decodes a "soft prompt" embedding back into a visual image.
        """
        self.eval()
        latents = torch.randn(
            (
                soft_prompt_embeds.shape[0],
                self.unet.config.in_channels,
                self.cfg.data.image_resolution // 8,
                self.cfg.data.image_resolution // 8,
            ),
            device=self.device,
            dtype=self.unet.dtype,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in tqdm(self.scheduler.timesteps, desc="Decoding Image"):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=soft_prompt_embeds
            ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        return image

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        **For Training:** Runs the full encode -> decode pipeline to compute training losses.
        """
        # --- 1. ENCODE: Image -> Soft Prompt ---
        soft_prompt_embeds, text_logits = self.encoder(
            images, self.cfg.training.gumbel_temperature
        )

        # --- 2. RECONSTRUCTION LOSS CALCULATION ---
        with torch.no_grad():
            latents = (
                self.vae.encode(images).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

        target_noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, target_noise, timesteps)

        predicted_noise = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=soft_prompt_embeds,
        ).sample

        # --- 3. CALCULATE LOSSES ---
        recon_loss = F.mse_loss(predicted_noise, target_noise)
        kl_loss = self._kl_divergence_loss(text_logits)

        total_loss = recon_loss + self.cfg.training.kl_weight * kl_loss

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
