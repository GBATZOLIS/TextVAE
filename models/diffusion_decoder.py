# In models/diffusion_decoder.py

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers import AutoencoderKL  # NEW: Import the VAE
import math


class DiffusionDecoder(nn.Module):
    """
    Reconstructs an image from a text representation using a conditional U-Net
    in the latent space.
    """

    def __init__(
        self, unet_model_name: str, scheduler_type: str = "ddpm", chunk_size: int = 75
    ):
        super().__init__()

        # --- 1. Load U-Net and Scheduler (as before) ---
        self.unet = UNet2DConditionModel.from_pretrained(
            unet_model_name, subfolder="unet"
        )
        if scheduler_type == "ddpm":
            self.scheduler = DDPMScheduler.from_pretrained(
                unet_model_name, subfolder="scheduler"
            )
        else:
            raise ValueError("Only DDPM scheduler is supported in this script.")

        # --- 2. NEW: Load the VAE ---
        self.vae = AutoencoderKL.from_pretrained(unet_model_name, subfolder="vae")

        # --- 3. NEW: Freeze VAE weights ---
        # We are not training the VAE, only using it for encoding/decoding.
        self.vae.requires_grad_(False)

        # --- 4. NEW: Define VAE scaling factor ---
        # This is a specific value used in Stable Diffusion to scale the latents
        # for better numerical stability during the diffusion process.
        self.vae_scale_factor = 0.18215

        self.chunk_size = chunk_size
        self.loss_fn = nn.MSELoss()

    def forward(
        self, soft_embeddings: torch.Tensor, original_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a single training step for the diffusion model in the latent space,
        ensuring all inputs to the U-Net are in the correct half-precision format.
        """
        # --- 1. VAE Encoding ---
        with torch.no_grad():
            latents = self.vae.encode(
                original_images.to(self.vae.dtype)
            ).latent_dist.sample()
        latents = latents * self.vae_scale_factor

        # --- 2. Diffusion Process ---
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # --- 3. Long Text Handling ---
        seq_len = soft_embeddings.shape[1]
        num_chunks = math.ceil(seq_len / self.chunk_size)
        padding_size = num_chunks * self.chunk_size - seq_len
        if padding_size > 0:
            # --- THIS IS THE CRITICAL FIX ---
            # Create padding with the same dtype and device as the embeddings
            padding = torch.zeros(
                soft_embeddings.shape[0],
                padding_size,
                soft_embeddings.shape[2],
                device=soft_embeddings.device,
                dtype=soft_embeddings.dtype,
            )
            soft_embeddings = torch.cat([soft_embeddings, padding], dim=1)

        chunked_embeddings = soft_embeddings.view(
            soft_embeddings.shape[0] * num_chunks, self.chunk_size, -1
        )

        # --- 4. Prepare U-Net Inputs ---
        unet_input = noisy_latents.half().repeat_interleave(num_chunks, dim=0)
        unet_timesteps = timesteps.repeat_interleave(num_chunks, dim=0)

        # --- 5. Predict Noise ---
        # Now ALL inputs (unet_input, chunked_embeddings) are float16
        predicted_noise = self.unet(
            unet_input, unet_timesteps, encoder_hidden_states=chunked_embeddings
        ).sample

        target_noise = noise.half().repeat_interleave(num_chunks, dim=0)

        loss = self.loss_fn(predicted_noise, target_noise)
        return loss
