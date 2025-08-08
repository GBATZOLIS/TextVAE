# models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import cast, List

from config import AppConfig
from .vision_encoder import VLMEncoder


class TextVAE(nn.Module):
    """
    The main text-based codec model.
    """

    def __init__(self, cfg: AppConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- 1. The VLM Encoder ---
        self.encoder = VLMEncoder(cfg, device)

        # --- 2. The Frozen Diffusion Decoder Components ---
        self.vae = AutoencoderKL.from_pretrained(
            cfg.models.t2i_id, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.models.t2i_id, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(
            cfg.models.t2i_id, subfolder="scheduler"
        )

        vlm_hidden_size = self.encoder.model.config.text_config.hidden_size  # 4096
        unet_cross_attn_dim = self.unet.config.cross_attention_dim  # 768
        self.projection = nn.Linear(vlm_hidden_size, unet_cross_attn_dim).to(
            device, dtype=torch.float16
        )

        # --- 3. The Text Encoder for the Decoder ---
        # We need this for the inference path (decode).
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.models.t2i_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.models.t2i_id, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)

        # --- 4. Freeze Components ---
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        # The VLMEncoder handles its own freezing logic internally

    def _kl_divergence_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates KL divergence against a uniform distribution to encourage diverse token usage."""
        log_q = F.log_softmax(logits, dim=-1)
        log_p = F.log_softmax(torch.ones_like(logits), dim=-1)
        return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)

    def _encode_long_prompt(self, prompt: str) -> torch.Tensor:
        """Encodes prompts longer than the 77 token limit by chunking and concatenating."""
        tokens = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=False
        ).input_ids.to(self.device)

        # Split tokens into chunks of 75 (77 - 2 for BOS/EOS)
        max_length = self.tokenizer.model_max_length - 2
        chunks = [
            tokens[0, i : i + max_length] for i in range(0, tokens.shape[1], max_length)
        ]

        # Pad each chunk and encode
        embeds = []
        for chunk in chunks:
            padded_chunk = self.tokenizer.build_inputs_with_special_tokens(
                chunk.tolist()
            )
            input_ids = torch.tensor([padded_chunk], device=self.device)
            embedding = self.text_encoder(input_ids).last_hidden_state
            embeds.append(embedding)

        return torch.cat(embeds, dim=1)

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> list[str]:
        """For Inference: Encodes an image into its human-readable text codebook entry."""
        self.eval()
        return cast(List[str], self.encoder.generate_text(image_tensor))

    @torch.no_grad()
    def decode(self, text_prompt: str, num_inference_steps: int = 50) -> torch.Tensor:
        """For Inference: Decodes a text prompt back into an image, handling long prompts."""
        self.eval()

        # Use the long prompt encoding strategy
        conditioning_embeds = self._encode_long_prompt(text_prompt)

        latents = torch.randn(
            (
                1,
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
                latent_model_input, t, encoder_hidden_states=conditioning_embeds
            ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        return image

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # --- 1. ENCODE: Image -> Soft Prompt ---
        soft_prompt_embeds, text_logits = self.encoder(images)
        if (
            torch.isnan(soft_prompt_embeds).any()
            or torch.isinf(soft_prompt_embeds).any()
        ):
            print("!!! Invalid values found in 'soft_prompt_embeds' from encoder !!!")

        # --- 2. PROJECT: Map to UNet's dimension ---
        # Pass the soft embeddings through our new projection layer.
        projected_embeds = self.projection(soft_prompt_embeds)
        if torch.isnan(projected_embeds).any() or torch.isinf(projected_embeds).any():
            print("!!! Invalid values found in 'projected_embeds' after projection !!!")

        # --- 3. RECONSTRUCTION LOSS ---
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

        # Use the *projected* embeddings for conditioning
        predicted_noise = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=projected_embeds,
        ).sample
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            print("!!! Invalid values found in 'predicted_noise' from UNet !!!")

        # --- 4. CALCULATE LOSSES ---
        recon_loss = F.mse_loss(predicted_noise, target_noise)
        kl_loss = self._kl_divergence_loss(text_logits)
        total_loss = recon_loss + self.cfg.training.kl_weight * kl_loss

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
