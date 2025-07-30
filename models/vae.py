# models/vae.py

"""High-level VAE model that binds encoder, decoder, Gumbel module and LLM prior."""
from __future__ import annotations
import logging
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Model
from config import VAEConfig
from .vision_encoder import VisionEncoder
from .text_decoder import TextDecoder
from .gumbel_softmax import GumbelSoftmax
from .diffusion.diffusion_decoder import DiffusionDecoder

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAE(nn.Module):
    """
    The main Variational Autoencoder model.

    This model architecture consists of:
    1. A VisionEncoder (ViT) to encode an image into feature space.
    2. A TextDecoder (Autoregressive Transformer) to generate discrete text tokens
       from the image features. This is the VAE's "encoder" in a sense (image -> tokens).
    3. A GumbelSoftmax module to allow for differentiable sampling of these tokens.
    4. A pre-trained Language Model (LLM) like GPT-2, which acts as a prior,
       guiding the generated tokens to be more language-like. The KL divergence
       is measured against this prior.
    5. A DiffusionDecoder (U-Net based) which reconstructs the image from the
       generated text tokens. This is the VAE's "decoder" (tokens -> image).

    Args:
        cfg (VAEConfig): Configuration object.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # --- Initialize Components ---
        self.vision_enc = VisionEncoder(cfg)
        self.text_dec = TextDecoder(cfg)
        self.gumbel = GumbelSoftmax(cfg)

        try:
            logger.info(f"Loading LLM prior: {cfg.llm_model_name}")
            self.llm = GPT2Model.from_pretrained(cfg.llm_model_name)
        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load LLM '{cfg.llm_model_name}'. Check model name and internet connection."
            )
            raise e

        if cfg.freeze_llm:
            logger.info("Freezing LLM parameters.")
            for p in self.llm.parameters():
                p.requires_grad_(False)
            self.llm.eval()  # Set to evaluation mode

        # The DiffusionDecoder shares token embeddings with the LLM prior.
        self.diffusion = DiffusionDecoder(cfg, self.llm.wte)

        # A learnable start-of-sequence token for the autoregressive text decoder.
        self.register_buffer("start_token", torch.zeros(1, 1, cfg.text_decoder_dim))
        nn.init.trunc_normal_(self.start_token, std=0.02)

    def encode(
        self, img: torch.Tensor, sample: bool = True
    ) -> Dict[str, torch.Tensor | None]:
        """
        Encodes an image into a sequence of discrete tokens.

        Args:
            img (torch.Tensor): Input image tensor. Shape: [B, C, H, W].
            sample (bool): If True, performs Gumbel-Softmax sampling. Otherwise, returns only logits.

        Returns:
            A dictionary containing:
            - "logits": Raw logits for each token position.
            - "soft_tokens": Differentiable, soft token representations.
            - "hard_tokens": Discrete token indices.
        """
        # 1. Get image features from the Vision Transformer.
        img_feat = self.vision_enc(img)
        B = img.size(0)

        logits, soft_tokens, hard_tokens = [], [], []

        # Initialize the generation with the start token.
        # This is the input to the TextDecoder at the first step.
        current_input_emb = self.start_token.expand(B, -1, -1)

        # 2. Autoregressively generate tokens one by one.
        for t in range(self.cfg.max_text_length):
            # Get logits for the *next* token from the text decoder.
            # We only care about the prediction for the last position in the sequence.
            output_logits = self.text_dec(current_input_emb, img_feat)[:, -1:, :]
            logits.append(output_logits)

            if not sample:
                continue

            # Sample the next token using Gumbel-Softmax.
            # `hard=True` uses the straight-through estimator.
            next_soft_token = self.gumbel(output_logits, hard=True)
            soft_tokens.append(next_soft_token)
            hard_tokens.append(next_soft_token.argmax(-1))

            # Embed the newly sampled token.
            next_token_emb = next_soft_token @ self.text_dec.token_emb.weight

            # Append the new token's embedding to the input sequence for the next step.
            current_input_emb = torch.cat([current_input_emb, next_token_emb], dim=1)

        # Concatenate results from all timesteps.
        results = {"logits": torch.cat(logits, dim=1)}
        if sample:
            results["soft_tokens"] = torch.cat(soft_tokens, dim=1)
            results["hard_tokens"] = torch.cat(hard_tokens, dim=1)
        else:
            results["soft_tokens"] = None
            results["hard_tokens"] = None

        return results

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodes a sequence of tokens back into an image using the diffusion model.
        This is a wrapper for the diffusion sampler.

        Args:
            tokens (torch.Tensor): The token sequence to condition on.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        logger.info("Starting decoding (image generation) from tokens.")
        return self.diffusion.sample(tokens)

    def compute_kl(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence between the model's token distribution (q)
        and the LLM prior's distribution (p). KL(q || p).

        Args:
            logits (torch.Tensor): The logits produced by the TextDecoder. Shape: [B, T, V].
            tokens (torch.Tensor): The sampled tokens (can be soft or hard). Shape: [B, T, V] or [B, T].

        Returns:
            torch.Tensor: The mean KL divergence over the batch.
        """
        # Ensure tokens are hard indices for querying the LLM.
        if tokens.dim() == 3:
            hard_tokens = tokens.argmax(-1)
        else:
            hard_tokens = tokens

        assert (
            hard_tokens.dim() == 2
        ), f"Tokens must be 2D for KL computation, but got {hard_tokens.dim()}"

        # Get the log-probability of the generated tokens under our model (q).
        log_q = F.log_softmax(logits, dim=-1)
        # Gather the log-probabilities of the specific tokens that were sampled.
        log_q_sel = torch.gather(log_q, -1, hard_tokens.unsqueeze(-1)).squeeze(-1)

        # Get the log-probability of the tokens under the frozen LLM prior (p).
        with torch.no_grad():
            # Get hidden states from the LLM.
            lm_out = self.llm(input_ids=hard_tokens)
            # Project hidden states to vocabulary logits.
            lm_logits = lm_out.last_hidden_state @ self.llm.wte.weight.T
            log_p = F.log_softmax(lm_logits, dim=-1)
            # Gather the log-probabilities of the specific tokens that were sampled.
            log_p_sel = torch.gather(log_p, -1, hard_tokens.unsqueeze(-1)).squeeze(-1)

        # KL divergence is the sum of (log_q - log_p) for each token in the sequence.
        # We then average this over the batch.
        kl_div = (log_q_sel - log_p_sel).sum(dim=1).mean()
        return kl_div

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The main forward pass for training.

        Args:
            img (torch.Tensor): A batch of input images.

        Returns:
            A dictionary containing the total loss and its components (recon, kl).
        """
        # 1. Encode image to tokens.
        enc_out = self.encode(img, sample=True)
        soft_tokens = enc_out["soft_tokens"]
        assert (
            soft_tokens is not None
        ), "Encoding failed to produce soft tokens during training."

        # 2. Compute the image reconstruction loss using the diffusion decoder.
        recon_loss = self.diffusion.p_losses(img, soft_tokens)

        # 3. Compute the KL divergence against the LLM prior.
        kl_loss = self.compute_kl(enc_out["logits"], soft_tokens)

        # 4. Compute the final VAE loss (ELBO).
        loss = recon_loss + self.cfg.kl_weight * kl_loss

        # 5. Anneal the Gumbel-Softmax temperature.
        self.gumbel.step()

        logger.info(
            f"Forward pass completed. Loss: {loss.item():.4f}, "
            f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}"
        )

        return {"loss": loss, "recon": recon_loss, "kl": kl_loss}


def build_vae_from_config(cfg: VAEConfig) -> TextVAE:
    """Factory function to build the VAE model from a config object."""
    logger.info(f"Building TextVAE model with seed: {cfg.seed}")
    torch.manual_seed(cfg.seed)
    # Potentially add CUDA seed if using GPU
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(cfg.seed)
    return TextVAE(cfg)
