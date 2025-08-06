# models/diffusion_decoder.py
import torch
import torch.nn as nn
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from controlnet_aux import CannyDetector
from PIL import Image


class DiffusionDecoder(nn.Module):
    """
    Decodes a prompt embedding into an image, conditioned by ControlNet.
    This module is used during training to calculate the reconstruction loss.
    """

    def __init__(self, t2i_id: str, controlnet_id: str, device: torch.device):
        super().__init__()
        self.device = device

        # --- Model Loading ---
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            t2i_id, controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)

        # --- Component Extraction ---
        # We extract the necessary components for a custom training loop.
        self.vae: AutoencoderKL = pipe.vae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        # --- ControlNet Preprocessor ---
        self.canny_detector = CannyDetector()

    def forward(
        self,
        soft_prompt_embeds: torch.Tensor,
        images_tensor: torch.Tensor,
        images_pil: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass for a single training step."""
        # 1. Encode images into the latent space using the VAE.
        with torch.no_grad():
            latents = (
                self.vae.encode(images_tensor).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

        # 2. Sample random noise and a random timestep for each image.
        target_noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()

        # 3. Create the noisy latents for the current timestep.
        noisy_latents = self.scheduler.add_noise(latents, target_noise, timesteps)

        # 4. Generate the ControlNet Canny edge map condition.
        with torch.no_grad():
            canny_maps = [
                self.canny_detector(img, low_threshold=100, high_threshold=200)
                for img in images_pil
            ]
            canny_condition = torch.cat(canny_maps).to(self.device, dtype=latents.dtype)

        # 5. Predict the noise using the UNet.
        predicted_noise = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=soft_prompt_embeds,
            down_block_additional_residuals=[canny_condition],
        ).sample

        return predicted_noise, target_noise
