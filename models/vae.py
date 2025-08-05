# models/vae.py

"""
Main VAE model for the proof-of-concept.

This version implements the "Hybrid Model" architecture. It trains an
encoder (CLIP -> TextDecoder) to generate a text code that instructs a
*frozen*, pre-trained Stable Diffusion model to reconstruct the image.

Key Architectural Changes:
1.  **Decoder is Pre-trained**: Uses Stable Diffusion's VAE and U-Net.
2.  **Operates in Latent Space**: The reconstruction loss is calculated on the
    difference between predicted noise and actual noise in Stable Diffusion's
    latent space, not pixel space.
3.  **Training Focus**: Only the vision-to-text components are trained.
4.  **Dual-Prior System**:
    - The **Reconstruction Prior** is CLIP's Text Encoder, providing the correct
      conditioning for the frozen Stable Diffusion U-Net.
    - The **Linguistic Prior** is GPT-2, used only to calculate the KL
      divergence and ensure the generated text is coherent.
"""
from __future__ import annotations
import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from tqdm import tqdm


from config import VAEConfig
from .vision_encoder import CLIPVisionEncoder
from .text_decoder import TextDecoder
from .gumbel_softmax import GumbelSoftmax

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAE(nn.Module):
    """
    The main VAE model, with corrected conditioning for the hybrid proof-of-concept.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # --- 1. Initialize Trainable Components ---
        self.vision_enc = CLIPVisionEncoder(cfg)
        self.gumbel = GumbelSoftmax(cfg)

        # --- 2. Initialize Frozen Priors and Decoders ---
        try:
            # Linguistic Prior (for KL divergence)
            logger.info(f"Loading LLM prior (for KL): {cfg.llm_prior_model}")
            self.llm_prior = GPT2Model.from_pretrained(cfg.llm_prior_model)
            self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.llm_prior_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Reconstruction Prior (for SD Conditioning)
            # This is the key fix: use the text encoder that SD was trained with.
            logger.info(
                f"Loading Text Encoder (for Reconstruction): {cfg.diffusion_decoder_model}"
            )
            self.text_encoder_for_recon = CLIPTextModel.from_pretrained(
                cfg.diffusion_decoder_model, subfolder="text_encoder"
            )
            # FIX: Load the correct tokenizer for the CLIP text encoder
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                cfg.diffusion_decoder_model, subfolder="tokenizer"
            )

            # Diffusion Decoder Components
            logger.info(f"Loading Diffusion Decoder: {cfg.diffusion_decoder_model}")
            self.diffusion_vae = AutoencoderKL.from_pretrained(
                cfg.diffusion_decoder_model, subfolder="vae"
            )
            self.diffusion_unet = UNet2DConditionModel.from_pretrained(
                cfg.diffusion_decoder_model, subfolder="unet"
            )

            self.scheduler = DDIMScheduler.from_pretrained(
                cfg.diffusion_decoder_model, subfolder="scheduler"
            )

        except Exception as e:
            logger.error(
                "Failed to load a pre-trained model. Check names and connection.",
                exc_info=True,
            )
            raise e

        # --- 3. Initialize the Trainable Text Decoder ---
        # The text decoder's vocabulary and embeddings MUST match the linguistic prior (GPT-2)
        # because its job is to generate coherent English text.
        self.text_dec = TextDecoder(
            cfg,
            token_embedding_layer=self.llm_prior.wte,
            vocab_size=self.llm_prior.config.vocab_size,
        )

        # --- Add a projection layer to bridge vision and text dimensions ---
        self.vision_feat_proj = nn.Linear(
            self.vision_enc.vision_tower.config.hidden_size,  # From CLIP Vision (e.g., 1024)
            cfg.text_decoder_dim,  # To Text Decoder (e.g., 768 for GPT-2)
        )

        # --- 4. Freeze All Pre-trained Components ---
        self.llm_prior.requires_grad_(False)
        self.llm_prior.eval()
        logger.info("Froze LLM prior (GPT-2) parameters.")

        self.text_encoder_for_recon.requires_grad_(False)
        self.text_encoder_for_recon.eval()
        logger.info("Froze Text Encoder (CLIP) parameters.")

        self.diffusion_vae.requires_grad_(False)
        self.diffusion_unet.requires_grad_(False)
        self.diffusion_vae.eval()
        self.diffusion_unet.eval()
        logger.info("Froze Diffusion Decoder (SD VAE + U-Net) parameters.")

        self.start_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else 50256
        )

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encodes an image into a sequence of discrete tokens using the trainable
        vision encoder and text decoder. This is run without gradients for inference.
        """
        self.eval()
        img_feat_raw = self.vision_enc(img)
        img_feat = self.vision_feat_proj(img_feat_raw)

        B = img.size(0)
        hard_tokens = torch.full(
            (B, 1), self.start_token_id, dtype=torch.long, device=img.device
        )

        for _ in range(self.cfg.max_text_length - 1):
            logits, _ = self.text_dec(hard_tokens, img_feat)
            next_token_logits = logits[:, -1, :]
            hard_next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            hard_tokens = torch.cat([hard_tokens, hard_next_token], dim=1)

        return {"hard_tokens": hard_tokens}

    @torch.no_grad()
    def decode(
        self, tokens: torch.Tensor, num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Decodes a sequence of tokens into an image using the frozen Stable Diffusion pipeline.
        This now uses the correct CLIP text encoder for conditioning.
        """
        self.eval()
        device = self.diffusion_unet.device

        # Get text embeddings from the CLIP text encoder
        encoder_hidden_states = self.text_encoder_for_recon(
            input_ids=tokens.to(device)
        ).last_hidden_state

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (
                tokens.shape[0],
                self.diffusion_unet.config.in_channels,
                self.cfg.image_size // 8,
                self.cfg.image_size // 8,
            ),
            device=device,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps, desc="Decoding Image"):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.diffusion_unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / self.diffusion_vae.config.scaling_factor * latents
        image = self.diffusion_vae.decode(latents).sample

        return image

    def compute_kl(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes KL divergence between the model's token distribution (q)
        and the frozen GPT-2 prior's distribution (p). This remains unchanged.
        """
        log_q = F.log_softmax(logits, dim=-1)
        # We need to handle padding tokens in the KL computation
        pad_mask = (tokens != self.tokenizer.pad_token_id).float()

        log_q_sel = torch.gather(log_q, -1, tokens.unsqueeze(-1)).squeeze(-1) * pad_mask

        with torch.no_grad():
            # Get logits from the frozen GPT-2 prior
            lm_out = self.llm_prior(input_ids=tokens).last_hidden_state
            # Use the shared lm_head to project GPT-2's output to the vocab space
            lm_logits = self.text_dec.lm_head(lm_out)
            log_p = F.log_softmax(lm_logits, dim=-1)
            log_p_sel = (
                torch.gather(log_p, -1, tokens.unsqueeze(-1)).squeeze(-1) * pad_mask
            )

        # Sum over the sequence length and average over the batch
        return (log_q_sel - log_p_sel).sum(dim=1).mean()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The main forward pass for training the hybrid model.
        """
        # --- 1. Encode Image to Latent Space using SD's VAE ---
        with torch.no_grad():
            latents = self.diffusion_vae.encode(img).latent_dist.sample()
            latents = latents * self.diffusion_vae.config.scaling_factor

        # --- 2. Encode Image to Text Code using our trainable encoder ---
        img_feat_raw = self.vision_enc(img)
        img_feat = self.vision_feat_proj(img_feat_raw)
        B = img.size(0)
        hard_tokens = torch.full(
            (B, 1), self.start_token_id, dtype=torch.long, device=img.device
        )
        all_logits = []

        for _ in range(self.cfg.max_text_length - 1):
            logits, _ = self.text_dec(hard_tokens, img_feat)
            next_token_logits = logits[:, -1, :]
            soft_next_token = self.gumbel(next_token_logits, hard=True)
            hard_next_token = soft_next_token.argmax(dim=-1, keepdim=True)
            hard_tokens = torch.cat([hard_tokens, hard_next_token], dim=1)
            all_logits.append(next_token_logits.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)

        # --- 3. Compute Reconstruction Loss in Latent Space ---
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (B,), device=latents.device
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # --- THE FIX: Bridge the vocabulary gap ---
        with torch.no_grad():
            # 3a. Decode the generated GPT-2 tokens into text strings
            text_descriptions = self.tokenizer.batch_decode(
                hard_tokens, skip_special_tokens=True
            )

            # 3b. Re-tokenize the text strings using the CLIP tokenizer
            clip_tokens = self.clip_tokenizer(
                text_descriptions,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(hard_tokens.device)

            # 3c. Get embeddings using the correct CLIP text encoder
            encoder_hidden_states = self.text_encoder_for_recon(
                input_ids=clip_tokens
            ).last_hidden_state

            # 3d. Predict noise using the U-Net
            noise_pred = self.diffusion_unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

        recon_loss = F.mse_loss(noise_pred, noise, reduction="mean")

        # --- 4. Compute KL Divergence against the GPT-2 Prior ---
        # This part remains the same, ensuring the generated text is coherent.
        kl_loss = self.compute_kl(logits, hard_tokens[:, 1:])

        # --- 5. Total Loss and Gumbel Annealing ---
        loss = recon_loss + self.cfg.kl_weight * kl_loss
        self.gumbel.step()

        return {"loss": loss, "recon": recon_loss, "kl": kl_loss}
