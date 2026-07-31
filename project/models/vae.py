"""§1 / §6 — the complete Image → Text → Image VAE.

``TextVAE`` wires the four components together and implements the training algorithm of
§6 verbatim:

1. ``H_img = ViT(x)``, then an autoregressive rollout of ``q_φ(z|x)`` with Gumbel-Softmax
   giving soft tokens ``y = (y_1, ..., y_T)``                              (§2, §3, §6.1)
2. ``Z' = (E_LM^T y_1, ..., E_LM^T y_T)``                                        (§5.1)
3. ``L_diff`` from the text-conditional DDPM                              (Eq. 7, §6.2)
4. ``L_KL = log q_φ(z|x) − log p(z)`` under the frozen LM prior           (Eq. 6, §6.3)
5. ``L = mean(L_diff) + β · mean(L_KL)``                                  (Eq. 8, §6.4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.losses import KLOutput, kl_divergence_mc
from ..utils.config import Config, LossConfig, ModelConfig
from .diffusion_decoder import DiffusionDecoder, DiffusionLossOutput
from .encoder import LatentSample, TextLatentEncoder
from .gumbel import TemperatureScheduler
from .prior import FrozenLanguageModelPrior

logger = logging.getLogger(__name__)

__all__ = ["TextVAE", "TextVAEOutput"]


@dataclass
class TextVAEOutput:
    """Every quantity of Eq. 8 for one batch, plus latents for logging/inspection."""

    loss: torch.Tensor  # Eq. 8 (scalar)
    diffusion_loss: torch.Tensor  # E_q[-log p_θ(x|z)] surrogate, Eq. 7 (scalar)
    kl: torch.Tensor  # β-weighted-free KL, mean over batch (scalar)
    log_q: torch.Tensor  # (B,) log q_φ(z|x)
    log_p: torch.Tensor  # (B,) log p(z)
    beta: float
    temperature: float
    latent: LatentSample
    kl_output: KLOutput
    diffusion_output: DiffusionLossOutput
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def elbo(self) -> torch.Tensor:
        """The (negated) ELBO of Eq. 1 as optimised here, i.e. ``L_diff + KL``."""
        return self.diffusion_loss + self.kl


class TextVAE(nn.Module):
    """The full model: ``q_φ(z|x)`` (§2) + frozen ``p(z)`` (§4) + ``p_θ(x|z)`` (§5)."""

    def __init__(
        self, model_config: ModelConfig, loss_config: Optional[LossConfig] = None
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config or LossConfig()

        # §4: the frozen prior is built first — it defines the latent vocabulary, the BOS
        # symbol and the embedding matrix E_LM that both other components must agree with.
        self.prior = FrozenLanguageModelPrior(model_config.prior)
        self.encoder = TextLatentEncoder(
            model_config.encoder,
            vocab_size=self.prior.vocab_size,
            bos_token_id=self.prior.bos_token_id,
            lm_name=model_config.prior.name,
        )
        self.decoder = DiffusionDecoder(
            model_config.decoder, context_dim=self.prior.hidden_size
        )
        self.temperature_scheduler = TemperatureScheduler.from_config(
            model_config.gumbel
        )

        logger.info(
            "TextVAE ready | T=%d |V|=%d | trainable: encoder %.1fM, decoder %.1fM",
            self.latent_length,
            self.prior.vocab_size,
            self.encoder.num_trainable_parameters() / 1e6,
            self.decoder.num_trainable_parameters() / 1e6,
        )

    @classmethod
    def from_config(cls, config: Config) -> "TextVAE":
        return cls(config.model, config.loss)

    # ------------------------------------------------------------------ properties
    @property
    def latent_length(self) -> int:
        return self.encoder.latent_length

    @property
    def vocab_size(self) -> int:
        return self.prior.vocab_size

    @property
    def device(self) -> torch.device:
        return self.prior.device

    # --------------------------------------------------------------------- forward
    def forward(
        self,
        images: torch.Tensor,
        temperature: Optional[float] = None,
        beta: Optional[float] = None,
        caption_input_ids: Optional[torch.Tensor] = None,
        stochastic: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> TextVAEOutput:
        """One full ELBO evaluation (§6 steps 1–4).

        Parameters
        ----------
        images:
            ``(B, 3, H, W)`` in ``[-1, 1]``.
        temperature:
            τ for Eq. 3; defaults to the scheduler's current value.
        beta:
            β of Eq. 8; defaults to ``loss_config.beta``.
        caption_input_ids:
            Optional captions for the auxiliary teacher-forced CE warm-start
            (``loss.aux_caption_ce_weight``; not part of the paper, default weight 0).
        """
        gumbel = self.model_config.gumbel
        tau = float(
            self.temperature_scheduler.current if temperature is None else temperature
        )
        beta_value = float(self.loss_config.beta if beta is None else beta)

        # --- §6 step 1: q_φ(z|x) rollout with Gumbel-Softmax -------------------------
        latent = self.encoder.sample_latent(
            images,
            temperature=tau,
            hard=gumbel.hard,
            straight_through=gumbel.straight_through,
            stochastic=stochastic,
            generator=generator,
        )

        # --- §5.1: Z' = E_LM^T y  (frozen LM embedding matrix) -----------------------
        text_embeddings = self.prior.embed_soft(latent.y)

        # --- §6 step 2: diffusion (reconstruction) loss, Eq. 7 -----------------------
        diffusion = self.decoder.diffusion_loss(
            images, text_embeddings, generator=generator
        )

        # --- §6 step 3: KL against the frozen prior, Eqs. 4-6 ------------------------
        estimator = self.loss_config.kl_estimator
        log_q = latent.log_q(estimator)
        prior_out = self.prior.log_prob(
            y=latent.y, token_ids=latent.token_ids, estimator=estimator
        )
        log_p = prior_out.total
        kl = kl_divergence_mc(
            log_q=log_q,
            log_p=log_p,
            reduction=self.loss_config.kl_reduction,
            latent_length=latent.latent_length,
            free_bits=self.loss_config.free_bits,
        )

        # --- §6 step 4: L = mean(L_diff) + β · mean(L_KL) ----------------------------
        aux_losses: Dict[str, torch.Tensor] = {}
        if self.loss_config.aux_caption_ce_weight > 0 and caption_input_ids is not None:
            aux_losses["caption_ce"] = (
                self.loss_config.aux_caption_ce_weight
                * self.caption_cross_entropy(images, caption_input_ids)
            )
        total = diffusion.loss + beta_value * kl.mean
        for value in aux_losses.values():
            total = total + value

        return TextVAEOutput(
            loss=total,
            diffusion_loss=diffusion.loss,
            kl=kl.mean,
            log_q=log_q,
            log_p=log_p,
            beta=beta_value,
            temperature=tau,
            latent=latent,
            kl_output=kl,
            diffusion_output=diffusion,
            aux_losses=aux_losses,
            metrics=self._metrics(
                latent, kl, log_q, log_p, prior_out.per_token, tau, beta_value
            ),
        )

    def _metrics(
        self,
        latent: LatentSample,
        kl: KLOutput,
        log_q: torch.Tensor,
        log_p: torch.Tensor,
        prior_per_token: torch.Tensor,
        tau: float,
        beta: float,
    ) -> Dict[str, float]:
        """Scalars worth watching: KL health, posterior sharpness, latent diversity."""
        with torch.no_grad():
            length = max(1, latent.latent_length)
            unique = torch.tensor(
                [
                    row.unique().numel() / length
                    for row in latent.token_ids.detach().cpu()
                ]
            ).mean()
            return {
                "kl": float(kl.mean),
                "kl_raw": float(kl.raw_per_sample.mean()),
                "kl_per_token": float(kl.raw_per_sample.mean() / length),
                "log_q": float(log_q.mean()),
                "log_p": float(log_p.mean()),
                "prior_nll_per_token": float(-prior_per_token.mean()),
                "prior_perplexity": float(torch.exp(-prior_per_token.mean())),
                "posterior_entropy": float(latent.entropy.mean()),
                "latent_unique_frac": float(unique),
                "temperature": tau,
                "beta": beta,
            }

    # ------------------------------------------------------------- auxiliary losses
    def caption_cross_entropy(
        self, images: torch.Tensor, caption_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced CE of real captions under ``q_φ`` (optional warm-start).

        Not part of the paper's objective: it only gives the encoder a head start at
        producing language-like sequences and is disabled by default
        (``loss.aux_caption_ce_weight = 0``).
        """
        logits = self.encoder(images, caption_input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            caption_input_ids.reshape(-1),
        )

    # ---------------------------------------------------------------- inference API
    def latent_texts(self, latent: LatentSample) -> List[str]:
        """Decode the latent token ids into human-readable strings."""
        return self.prior.decode(latent.token_ids)

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        temperature: Optional[float] = None,
        stochastic: bool = False,
    ) -> Tuple[LatentSample, List[str]]:
        """Image → latent text. Returns the sample and its decoded strings."""
        tau = float(
            self.model_config.gumbel.eval_temperature
            if temperature is None
            else temperature
        )
        latent = self.encoder.sample_latent(
            images,
            temperature=tau,
            hard=True,
            straight_through=False,
            stochastic=stochastic,
        )
        return latent, self.latent_texts(latent)

    @torch.no_grad()
    def decode(
        self,
        latent_y: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Latent text (one-hot/soft ``y``) → image, via the reverse diffusion process."""
        text_embeddings = self.prior.embed_soft(latent_y)
        return self.decoder.sample(
            text_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_size=image_size,
            generator=generator,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        images: torch.Tensor,
        temperature: Optional[float] = None,
        stochastic: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, List[str], LatentSample]:
        """Full image → text → image round trip. Returns (images, latent texts, latent)."""
        latent, texts = self.encode(
            images, temperature=temperature, stochastic=stochastic
        )
        recon = self.decode(
            self.prior.one_hot(latent.token_ids),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
            generator=generator,
        )
        return recon, texts, latent

    @torch.no_grad()
    def sample_from_prior(
        self,
        num_samples: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Unconditional generation: ``z ~ p(z)`` from the LM, then ``x ~ p_θ(x|z)``."""
        token_ids = self.prior.sample(
            num_samples,
            length=self.latent_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        images = self.decode(
            self.prior.one_hot(token_ids),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_size=image_size,
            generator=generator,
        )
        return images, self.prior.decode(token_ids)

    @torch.no_grad()
    def decode_from_text(
        self,
        texts: Sequence[str],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Hand-written text → image, i.e. driving the decoder through the latent space."""
        token_ids = self.prior.encode_text(texts, length=self.latent_length)
        return self.decode(
            self.prior.one_hot(token_ids),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_size=image_size,
            generator=generator,
        )

    # ---------------------------------------------------------------- checkpointing
    def frozen_key_prefixes(self) -> Tuple[str, ...]:
        """State-dict prefixes that are skipped when checkpointing.

        A module is skipped only when it is both (a) frozen, so it never changes during
        training, and (b) *reproducible from the config* — i.e. loaded from named
        pretrained weights. That keeps checkpoints small (the prior alone is hundreds of
        MB) without ever losing state: a randomly initialised frozen module is
        run-specific and therefore always saved.
        """
        prefixes: List[str] = []
        if self.model_config.prior.pretrained:
            prefixes.append("prior.")
        vision = self.model_config.encoder.vision
        if vision.freeze_backbone and vision.pretrained:
            prefixes.append("encoder.vision_encoder.")
        if self.decoder.image_vae is not None:
            prefixes.append("decoder.image_vae.")  # always from_pretrained
        return tuple(prefixes)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Trainable state only (see :meth:`frozen_key_prefixes`)."""
        state = super().state_dict(*args, **kwargs)
        frozen = self.frozen_key_prefixes()
        return {k: v for k, v in state.items() if not k.startswith(frozen)}

    def load_state_dict(
        self, state_dict: Any, strict: bool = True, assign: bool = False
    ) -> Any:
        """Load a filtered checkpoint, keeping ``strict`` semantics for trainable keys.

        The frozen entries omitted by :meth:`state_dict` are back-filled from the freshly
        built modules, so a genuinely missing *trainable* key is still an error.
        """
        merged = dict(state_dict)
        frozen = self.frozen_key_prefixes()
        for key, value in super().state_dict().items():
            if key not in merged and key.startswith(frozen):
                merged[key] = value
        return super().load_state_dict(merged, strict=strict, assign=assign)

    # -------------------------------------------------------------------- utilities
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def parameter_groups(
        self, encoder_lr: Optional[float] = None, decoder_lr: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Optimiser groups for ``φ`` (encoder) and ``θ`` (decoder); the prior is absent."""
        groups: List[Dict[str, Any]] = []
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        if encoder_params:
            group: Dict[str, Any] = {"params": encoder_params, "name": "encoder"}
            if encoder_lr is not None:
                group["lr"] = encoder_lr
            groups.append(group)
        if decoder_params:
            group = {"params": decoder_params, "name": "decoder"}
            if decoder_lr is not None:
                group["lr"] = decoder_lr
            groups.append(group)
        return groups
