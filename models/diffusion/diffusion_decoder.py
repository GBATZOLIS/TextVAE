# models/diffusion/diffusion_decoder.py

"""DDPM wrapper that ties UNet with beta schedule and sampling utilities."""
from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig
from .unet import UNet

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DiffusionDecoder(nn.Module):
    """
    Manages the diffusion and reverse diffusion (sampling) processes.

    This module defines the forward process (q_sample) which adds noise to data,
    and the reverse process (p_losses, sample) which learns to remove it. It uses
    a pre-computed beta schedule to control the noise level at each timestep.

    Args:
        cfg (VAEConfig): Configuration object with model parameters.
        embedding_layer (nn.Embedding): The token embedding layer, shared with
                                        the language model prior.
    """

    def __init__(self, cfg: VAEConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        # The embedding layer is shared with the LLM to align the latent space
        self.embed = embedding_layer
        self.net = UNet(cfg)

        # --- Pre-compute diffusion schedule constants ---
        # The beta schedule determines the variance of noise added at each timestep.
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_timesteps)
        alphas = 1.0 - betas
        # alphas_cumprod is the cumulative product of alphas, used for direct sampling at any timestep t.
        alphas_cum = torch.cumprod(alphas, dim=0)

        # Registering as buffers saves them with the model state_dict without being considered parameters.
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_acum", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_acum", torch.sqrt(1.0 - alphas_cum))

    def token_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embeds tokens using the shared embedding layer.

        Handles both hard integer tokens (shape: [B, T]) and soft,
        differentiable token distributions (shape: [B, T, V]).

        Args:
            tokens (torch.Tensor): Input tokens, either hard or soft.

        Returns:
            torch.Tensor: The corresponding token embeddings.
        """
        # Hard tokens (indices) are passed through the embedding layer directly.
        if tokens.dim() == 2:
            return self.embed(tokens)
        # Soft tokens (probabilities) are embedded via a matrix multiplication with the embedding weight.
        # This is a differentiable way to get weighted-average embeddings.
        if tokens.dim() == 3:
            return tokens @ self.embed.weight

        raise ValueError(f"Unsupported token tensor dimensionality: {tokens.dim()}")

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward diffusion process (q). Adds noise to an image x0 at timestep t.

        This uses the reparameterization trick: x_t = sqrt(alpha_cum_t)*x_0 + sqrt(1-alpha_cum_t)*noise.
        It allows sampling x_t directly without iterating through all previous timesteps.

        Args:
            x0 (torch.Tensor): The initial clean image (t=0). Shape: [B, C, H, W].
            t (torch.Tensor): Timestep indices. Shape: [B].
            noise (torch.Tensor): Standard Gaussian noise. Shape: [B, C, H, W].

        Returns:
            torch.Tensor: The noised image at timestep t.
        """
        # Gather the pre-computed schedule values for the given timesteps t.
        # [:, None, None, None] reshapes for broadcasting over the image dimensions.
        sqrt_alpha_cum_t = self.sqrt_acum[t, None, None, None]
        sqrt_one_minus_alpha_cum_t = self.sqrt_one_minus_acum[t, None, None, None]

        # Apply the forward process formula.
        return sqrt_alpha_cum_t * x0 + sqrt_one_minus_alpha_cum_t * noise

    def p_losses(self, x0: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes the training loss for the diffusion model.

        The model is trained to predict the noise that was added to the image.

        Args:
            x0 (torch.Tensor): The initial clean image. Shape: [B, C, H, W].
            tokens (torch.Tensor): Conditional text tokens. Shape: [B, T] or [B, T, V].

        Returns:
            torch.Tensor: The mean squared error loss between predicted noise and actual noise.
        """
        B = x0.size(0)
        # 1. Sample a random timestep t for each image in the batch.
        t = torch.randint(
            0, self.cfg.diffusion_timesteps, (B,), device=x0.device, dtype=torch.long
        )

        # 2. Sample Gaussian noise and create the noised image x_t.
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # 3. Get text embeddings to condition the UNet.
        txt_cond = self.token_embed(tokens)

        # 4. Predict the noise from the noised image.
        predicted_noise = self.net(x_t, t, txt_cond)

        # 5. Calculate the MSE loss.
        loss = F.mse_loss(predicted_noise, noise)
        logger.debug(f"Diffusion p_losses - Batch Size: {B}, Loss: {loss.item():.4f}")
        return loss

    @torch.no_grad()
    def sample(
        self, tokens: torch.Tensor, shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """
        Generates an image from noise by reversing the diffusion process (DDPM sampling).

        Starts with pure noise (x_T) and iteratively denoises it over T timesteps.

        Args:
            tokens (torch.Tensor): Conditional text tokens.
            shape (tuple, optional): The desired output shape (B, C, H, W).
                                     If None, uses default from config.

        Returns:
            torch.Tensor: The generated image, clamped to [-1, 1].
        """
        B = tokens.size(0)
        # If no shape is provided, construct it from the config.
        if shape is None:
            shape = (
                B,
                self.cfg.image_channels,
                self.cfg.image_size,
                self.cfg.image_size,
            )
        else:
            # Validate provided shape
            assert shape[0] == B, "Batch size in shape must match tokens batch size."
            assert len(shape) == 4, "Shape must be in (B, C, H, W) format."

        # Start with pure Gaussian noise (x_T).
        x = torch.randn(shape, device=tokens.device)

        # Get text conditioning embeddings once.
        txt_cond = self.token_embed(tokens)
        logger.info(
            f"Starting sampling for batch size {B} over {self.cfg.diffusion_timesteps} timesteps."
        )

        # The reverse diffusion loop from T-1 down to 0.
        for t in reversed(range(self.cfg.diffusion_timesteps)):
            logger.debug(f"Sampling step t={t}")

            # Create a tensor of the current timestep for the whole batch.
            t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)

            # Predict noise using the UNet.
            pred_noise = self.net(x, t_tensor, txt_cond)

            # Get schedule constants for timestep t.
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_acum_t = self.sqrt_one_minus_acum[t]

            # The DDPM sampling formula to compute x_{t-1} from x_t.
            term1 = 1 / sqrt_alpha_t
            term2 = beta_t / sqrt_one_minus_acum_t
            x = term1 * (x - term2 * pred_noise)

            # Add noise back in, except for the last step.
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        logger.info("Sampling complete.")
        # Clamp the final image to the valid [-1, 1] range.
        return x.clamp(-1, 1)
