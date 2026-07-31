"""§5 — the decoder ``p_θ(x | z)``: a text-conditional DDPM.

* §5.2(1) forward process: ``x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε`` (diffusers' ``DDPMScheduler``,
  verified against the closed form in ``tests/test_diffusion_decoder.py``).
* §5.2(2) reverse process: ``ε_θ`` is a Stable-Diffusion-style ``UNet2DConditionModel``
  whose cross-attention layers consume the latent text embeddings
  ``Z' = (E_LM^T y_1, ..., E_LM^T y_T)`` from §5.1.
* Eq. 7: ``L_diff = E_{t,x_0,ε} ‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t, Z')‖²``.

Two image spaces are supported (see plan ambiguity #6/§3.3):

``pixel``
    The U-Net denoises ``x_0`` directly, exactly as Eq. 7 is written. Default.
``vae``
    A frozen ``AutoencoderKL`` maps ``x_0`` into a latent space first (latent diffusion).
    This is a deliberate, documented deviation that makes ≥256px training feasible; the
    objective is unchanged apart from operating on VAE latents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import DiffusionDecoderConfig

if TYPE_CHECKING:  # imported lazily at runtime; needed here only for annotations
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

logger = logging.getLogger(__name__)

__all__ = ["DiffusionDecoder", "DiffusionLossOutput"]


@dataclass
class DiffusionLossOutput:
    """Eq. 7 for one batch."""

    loss: torch.Tensor  # scalar, mean over the batch
    per_sample: torch.Tensor  # (B,) per-image MSE (post weighting)
    timesteps: torch.Tensor  # (B,) sampled t
    prediction: torch.Tensor  # ε_θ output
    target: torch.Tensor  # ε (or the velocity for v-prediction)


class DiffusionDecoder(nn.Module):
    """``p_θ(x | z)`` — noise-prediction network conditioned on the latent text.

    Parameters
    ----------
    config:
        :class:`~project.utils.config.DiffusionDecoderConfig`.
    context_dim:
        ``d_model`` of the frozen LM, i.e. the width of ``Z'`` (§5.1).
    """

    def __init__(self, config: DiffusionDecoderConfig, context_dim: int) -> None:
        super().__init__()
        from diffusers import DDPMScheduler

        self.config = config
        self.context_dim = int(context_dim)

        self.unet: "UNet2DConditionModel" = self._build_unet()
        self.cross_attention_dim = int(self.unet.config.cross_attention_dim)
        self.context_proj = self._build_context_projection()
        # "Unconditional" context for classifier-free guidance / conditioning dropout,
        # broadcast over the sequence dimension. It is only *learned* when conditioning
        # dropout is enabled; otherwise it stays a fixed zero embedding, so it never
        # becomes a parameter that receives no gradient (which would also break DDP).
        null_context = torch.zeros(1, 1, self.cross_attention_dim)
        if config.cond_dropout_prob > 0:
            self.null_context = nn.Parameter(null_context)
        else:
            self.register_buffer("null_context", null_context, persistent=True)

        self.scheduler: "DDPMScheduler" = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
        )

        self.image_vae: Optional["AutoencoderKL"] = None
        if config.image_space == "vae":
            self.image_vae = self._build_image_vae()

        if config.unet.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    # ---------------------------------------------------------------- construction
    def _build_unet(self) -> "UNet2DConditionModel":
        from diffusers import UNet2DConditionModel

        spec = self.config.unet
        if spec.pretrained:
            logger.info("Loading pretrained UNet %s/%s.", spec.name, spec.subfolder)
            unet = UNet2DConditionModel.from_pretrained(
                spec.name, subfolder=spec.subfolder or None
            )
        else:
            unet = UNet2DConditionModel(
                sample_size=spec.sample_size,
                in_channels=spec.in_channels,
                out_channels=spec.out_channels,
                block_out_channels=tuple(spec.block_out_channels),
                layers_per_block=spec.layers_per_block,
                down_block_types=tuple(spec.down_block_types),
                up_block_types=tuple(spec.up_block_types),
                cross_attention_dim=spec.cross_attention_dim,
                attention_head_dim=spec.attention_head_dim,
                norm_num_groups=spec.norm_num_groups,
            )
        if spec.lora.enabled:
            unet = self._apply_lora(unet, spec)
        return unet

    @staticmethod
    def _apply_lora(unet: "UNet2DConditionModel", spec: Any) -> "UNet2DConditionModel":
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        lora = spec.lora
        peft_config = PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules or ["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        logger.info("Applied LoRA (r=%d) to the UNet.", lora.r)
        return cast(
            "UNet2DConditionModel", get_peft_model(cast(Any, unet), peft_config)
        )

    def _build_context_projection(self) -> nn.Module:
        """Project ``Z'`` (width ``d_LM``) to the U-Net's cross-attention width.

        Documented ambiguity: the paper feeds ``Z'`` straight into cross-attention, which
        presumes ``d_LM == cross_attention_dim``. A pretrained U-Net fixes the latter, so a
        single trainable linear map is used unless the widths already agree and
        ``context_projection: identity`` is configured.
        """
        if self.config.context_projection == "identity":
            if self.context_dim != self.cross_attention_dim:
                raise ValueError(
                    f"context_projection='identity' requires d_LM ({self.context_dim}) == "
                    f"cross_attention_dim ({self.cross_attention_dim})"
                )
            return nn.Identity()
        return nn.Linear(self.context_dim, self.cross_attention_dim)

    def _build_image_vae(self) -> "AutoencoderKL":
        from diffusers import AutoencoderKL

        logger.info(
            "Loading frozen image VAE %s for latent diffusion.", self.config.vae_name
        )
        vae = AutoencoderKL.from_pretrained(
            self.config.vae_name, subfolder=self.config.vae_subfolder or None
        )
        vae.requires_grad_(False)
        vae.eval()
        latent_channels = int(vae.config.latent_channels)
        if latent_channels != int(self.unet.config.in_channels):
            raise ValueError(
                f"UNet in_channels ({self.unet.config.in_channels}) must match the image "
                f"VAE latent channels ({latent_channels}) in image_space='vae'"
            )
        return vae

    # ------------------------------------------------------------------ properties
    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def num_train_timesteps(self) -> int:
        """``T_diff``."""
        return int(self.scheduler.config.num_train_timesteps)

    @property
    def downsample_factor(self) -> int:
        """Spatial factor the U-Net requires the input to be divisible by."""
        return 2 ** (len(self.unet.config.block_out_channels) - 1)

    def train(self, mode: bool = True) -> "DiffusionDecoder":
        super().train(mode)
        if self.image_vae is not None:
            self.image_vae.eval()  # frozen, always eval
        return self

    # ------------------------------------------------------------- image <-> latent
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``x_0`` for the diffusion process: the image itself, or its VAE latent."""
        if self.image_vae is None:
            return images
        with torch.no_grad():
            posterior = self.image_vae.encode(
                images.to(self.image_vae.dtype)
            ).latent_dist
            latents = posterior.sample() * self.image_vae.config.scaling_factor
        return latents.to(images.dtype)

    @torch.no_grad()
    def decode_to_images(self, x0: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`encode_images`; returns images in ``[-1, 1]``."""
        if self.image_vae is None:
            return x0.clamp(-1, 1)
        latents = x0.to(self.image_vae.dtype) / self.image_vae.config.scaling_factor
        return self.image_vae.decode(latents).sample.clamp(-1, 1)

    # ------------------------------------------------------------------ conditioning
    def prepare_context(
        self,
        text_embeddings: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        cond_dropout: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Project ``Z'`` and optionally drop conditioning (for CFG training)."""
        if text_embeddings.dim() != 3:
            raise ValueError(
                f"expected (B, T, d_LM) latent text embeddings, got {tuple(text_embeddings.shape)}"
            )
        if text_embeddings.shape[-1] != self.context_dim:
            raise ValueError(
                f"latent text width {text_embeddings.shape[-1]} != configured d_LM {self.context_dim}"
            )
        context = self.context_proj(text_embeddings.to(self.dtype))
        if cond_dropout and self.config.cond_dropout_prob > 0:
            batch = context.shape[0]
            drop = (
                torch.rand(batch, device=context.device, generator=generator)
                < self.config.cond_dropout_prob
            )
            null = self.null_context.expand_as(context)
            context = torch.where(drop[:, None, None], null, context)
        return context

    def unconditional_context(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return self.null_context.expand(batch_size, seq_len, self.cross_attention_dim)

    # ------------------------------------------------------------ diffusion process
    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """§5.2(1): ``x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε``."""
        return self.scheduler.add_noise(x0, noise, timesteps)

    def predict_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``ε_θ(x_t, t, Z')`` — the U-Net's cross-attention consumes ``Z'``."""
        prediction = cast(Any, self.unet)(
            noisy,
            timesteps,
            encoder_hidden_states=context,
            encoder_attention_mask=context_mask,
        ).sample
        if prediction.shape[1] == 2 * noisy.shape[1]:
            # Some checkpoints predict (ε, Σ); the paper's objective only uses ε.
            prediction, _ = prediction.chunk(2, dim=1)
        return prediction

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
            dtype=torch.long,
        )

    def _snr_weights(self, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional min-SNR-γ reweighting (off by default; Eq. 7 has uniform weights)."""
        gamma = self.config.min_snr_gamma
        if gamma is None:
            return None
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha = alphas_cumprod[timesteps]
        snr = alpha / (1.0 - alpha)
        return torch.clamp(snr, max=gamma) / snr

    def diffusion_loss(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> DiffusionLossOutput:
        """Eq. 7 — the reconstruction term of the ELBO.

        ``images`` are in ``[-1, 1]``; ``text_embeddings`` is ``Z'`` of shape
        ``(B, T, d_LM)``. ``timesteps``/``noise`` may be supplied for deterministic tests.
        """
        if images.dim() != 4:
            raise ValueError(f"expected (B, C, H, W) images, got {tuple(images.shape)}")
        x0 = self.encode_images(images.to(self.dtype))
        factor = self.downsample_factor
        if x0.shape[-1] % factor or x0.shape[-2] % factor:
            raise ValueError(
                f"spatial size {tuple(x0.shape[-2:])} must be divisible by {factor} "
                "(UNet downsampling factor)"
            )

        context = self.prepare_context(
            text_embeddings,
            context_mask,
            cond_dropout=self.training,
            generator=generator,
        )
        if timesteps is None:
            timesteps = self._sample_timesteps(x0.shape[0], x0.device, generator)
        if noise is None:
            noise = torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )

        noisy = self.add_noise(x0, noise, timesteps)
        prediction = self.predict_noise(noisy, timesteps, context, context_mask)

        if self.config.prediction_type == "epsilon":
            target = noise
        else:  # v_prediction
            target = self.scheduler.get_velocity(x0, noise, timesteps)

        per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none")
        per_sample = per_sample.flatten(1).mean(dim=1)
        weights = self._snr_weights(timesteps)
        if weights is not None:
            per_sample = per_sample * weights
        return DiffusionLossOutput(
            loss=per_sample.mean(),
            per_sample=per_sample,
            timesteps=timesteps,
            prediction=prediction,
            target=target,
        )

    # ------------------------------------------------------------------- generation
    def _inference_scheduler(self) -> Any:
        from diffusers import DDIMScheduler, DDPMScheduler

        if self.config.inference_scheduler == "ddim":
            # DDPM's ancestral sampler needs ~1000 steps; DDIM gives usable samples in
            # tens of steps from the very same trained ε_θ.
            return DDIMScheduler.from_config(self.scheduler.config)
        return DDPMScheduler.from_config(self.scheduler.config)

    @torch.no_grad()
    def sample(
        self,
        text_embeddings: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run the reverse process to reconstruct an image from a latent text sequence.

        Returns images in ``[-1, 1]`` of shape ``(B, 3, H, W)``. With
        ``return_intermediates`` a list of intermediate decodes is returned instead
        (useful for the qualitative reports produced by ``scripts/inference.py``).
        """
        steps = int(num_inference_steps or self.config.num_inference_steps)
        scale = float(
            self.config.guidance_scale if guidance_scale is None else guidance_scale
        )
        scheduler = self._inference_scheduler()
        scheduler.set_timesteps(steps, device=self.device)

        context = self.prepare_context(
            text_embeddings, context_mask, cond_dropout=False
        )
        batch, seq_len = context.shape[0], context.shape[1]

        channels = int(self.unet.config.in_channels)
        if image_size is None:
            size = int(self.unet.config.sample_size or 64)
            spatial: Tuple[int, int] = (size, size)
        elif self.image_vae is not None:
            downscale = 2 ** (len(self.image_vae.config.block_out_channels) - 1)
            spatial = (image_size[0] // downscale, image_size[1] // downscale)
        else:
            spatial = image_size

        latents = torch.randn(
            (batch, channels, *spatial),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        latents = latents * scheduler.init_noise_sigma

        use_cfg = scale > 1.0
        if use_cfg:
            null = self.unconditional_context(batch, seq_len)
            context_in = torch.cat([null, context], dim=0)
            mask_in = (
                None if context_mask is None else torch.cat([context_mask] * 2, dim=0)
            )
        else:
            context_in, mask_in = context, context_mask

        intermediates: List[torch.Tensor] = []
        for t in scheduler.timesteps:
            model_input = torch.cat([latents] * 2) if use_cfg else latents
            model_input = scheduler.scale_model_input(model_input, t)
            timesteps = t.expand(model_input.shape[0]).to(self.device)
            prediction = self.predict_noise(model_input, timesteps, context_in, mask_in)
            if use_cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + scale * (cond - uncond)
            latents = scheduler.step(
                prediction, t, latents, generator=generator
            ).prev_sample
            if return_intermediates:
                intermediates.append(self.decode_to_images(latents))

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return self.decode_to_images(latents)

    # -------------------------------------------------------------------- utilities
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"image_space={self.config.image_space!r}, T_diff={self.num_train_timesteps}, "
            f"prediction_type={self.config.prediction_type!r}, d_LM={self.context_dim} -> "
            f"cross_attention_dim={self.cross_attention_dim}"
        )
