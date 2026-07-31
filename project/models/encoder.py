"""§2 — the encoder ``q_φ(z | x) = Π_t q_φ(z_t | z_<t, x)``.

Composition:

    x ──► ViT ──► H_img ──► projection ──► (cross-attention memory)
                                              │
    BOS ──► causal decoder ──► l_t ──► Gumbel-Softmax ──► y_t ──► E^T y_t ──┘  (next input)

The latent is sampled by a genuine autoregressive rollout (§6 step 1): the VAE is
unsupervised, so there is no caption to teacher-force on — each step conditions on the
model's own previous (soft) tokens. The loop is differentiable end to end, so gradients
from both the diffusion loss and the KL term reach ``φ`` through the Gumbel-Softmax
relaxation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.config import EncoderConfig
from .gumbel import gumbel_softmax_sample
from .text_decoder import TextDecoderBase, build_text_decoder
from .vision_encoder import VisionEncoder

logger = logging.getLogger(__name__)

__all__ = ["TextLatentEncoder", "LatentSample", "VisualProjector"]


@dataclass
class LatentSample:
    """One Monte Carlo draw ``z ~ q_φ(z | x)`` and everything the ELBO needs from it.

    Attributes
    ----------
    logits:
        ``(l_1, ..., l_T)`` of shape ``(B, T, |V|)`` — the encoder output of §2.1.
    y:
        The Gumbel-Softmax sample (hard/straight-through or soft), ``(B, T, |V|)``.
    y_soft:
        The purely relaxed Eq. 3 sample, ``(B, T, |V|)``.
    token_ids:
        Discrete latent tokens ``argmax_k y_{t,k}``, ``(B, T)``.
    token_log_q_relaxed:
        ``Σ_k y_{t,k} log q_φ(z_t = k | z_<t, x)`` — the relaxed score of Eq. 2.
    token_log_q_hard:
        ``log q_φ(z_t = argmax | z_<t, x)`` — the discrete score of Eq. 2.
    entropy:
        ``H[q_φ(z_t | z_<t, x)]`` per position, for monitoring posterior collapse.
    temperature:
        The τ used for this draw.
    """

    logits: torch.Tensor
    y: torch.Tensor
    y_soft: torch.Tensor
    token_ids: torch.Tensor
    token_log_q_relaxed: torch.Tensor
    token_log_q_hard: torch.Tensor
    entropy: torch.Tensor
    temperature: float

    @property
    def log_q_relaxed(self) -> torch.Tensor:
        """``log q_φ(z | x)`` for the relaxed sample, shape ``(B,)``."""
        return self.token_log_q_relaxed.sum(dim=-1)

    @property
    def log_q_hard(self) -> torch.Tensor:
        """``log q_φ(z | x)`` for the discrete sample, shape ``(B,)``."""
        return self.token_log_q_hard.sum(dim=-1)

    def log_q(self, estimator: str = "relaxed") -> torch.Tensor:
        if estimator == "relaxed":
            return self.log_q_relaxed
        if estimator == "hard":
            return self.log_q_hard
        raise ValueError(f"unknown KL estimator {estimator!r}")

    def token_log_q(self, estimator: str = "relaxed") -> torch.Tensor:
        if estimator == "relaxed":
            return self.token_log_q_relaxed
        if estimator == "hard":
            return self.token_log_q_hard
        raise ValueError(f"unknown KL estimator {estimator!r}")

    @property
    def latent_length(self) -> int:
        return int(self.token_ids.shape[1])

    def detach(self) -> "LatentSample":
        return LatentSample(
            logits=self.logits.detach(),
            y=self.y.detach(),
            y_soft=self.y_soft.detach(),
            token_ids=self.token_ids.detach(),
            token_log_q_relaxed=self.token_log_q_relaxed.detach(),
            token_log_q_hard=self.token_log_q_hard.detach(),
            entropy=self.entropy.detach(),
            temperature=self.temperature,
        )


class VisualProjector(nn.Module):
    """Maps ``H_img ∈ R^{N×d_vis}`` to the text decoder's width ``d_model``.

    The paper writes ``d_model`` for both; in practice a pretrained ViT and a pretrained
    GPT-2 have different widths, so this (trainable) projection is required. With
    ``num_layers=1`` it is a plain linear map.
    """

    def __init__(
        self, in_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("projector_layers must be >= 1")
        if num_layers == 1:
            self.proj: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            layers: List[nn.Module] = [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ]
            for _ in range(num_layers - 2):
                layers += [
                    nn.Linear(out_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                ]
            layers += [nn.Dropout(dropout), nn.Linear(out_dim, out_dim)]
            self.proj = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class TextLatentEncoder(nn.Module):
    """``q_φ(z | x)``: a vision encoder plus an autoregressive text decoder.

    Parameters
    ----------
    config:
        :class:`~project.utils.config.EncoderConfig`.
    vocab_size:
        ``|V|`` — must equal the prior's vocabulary (they share a tokenizer).
    bos_token_id:
        Start symbol for the rollout; pass the prior's so that ``q`` and ``p`` score
        ``z_1`` under identical context.
    lm_name:
        Pretrained LM whose weights initialise the text decoder (``gpt2`` backend).
    """

    def __init__(
        self,
        config: EncoderConfig,
        vocab_size: int,
        bos_token_id: int,
        lm_name: str = "gpt2",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.bos_token_id = int(bos_token_id)
        self.latent_length = int(config.latent_length)

        self.vision_encoder = VisionEncoder(config.vision)
        self.text_decoder: TextDecoderBase = build_text_decoder(
            config.text_decoder, vocab_size=vocab_size, lm_name=lm_name
        )
        self.projector = VisualProjector(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=self.text_decoder.d_model,
            num_layers=config.projector_layers,
            dropout=config.projector_dropout,
        )
        logger.info(
            "Encoder ready: d_vis=%d -> d_model=%d, T=%d, |V|=%d",
            self.vision_encoder.hidden_size,
            self.text_decoder.d_model,
            self.latent_length,
            self.vocab_size,
        )

    # ------------------------------------------------------------------ image side
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``H_img = proj(ViT(x))`` plus its attention mask (§2.1 step 1)."""
        vision_out = self.vision_encoder(images)
        memory = self.projector(vision_out.hidden_states.to(self.projector_dtype))
        return memory, vision_out.attention_mask

    @property
    def projector_dtype(self) -> torch.dtype:
        return next(self.projector.parameters()).dtype

    # -------------------------------------------------------------- teacher forcing
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced logits for a *given* token sequence, shape ``(B, L, |V|)``.

        Used by the optional auxiliary caption-CE warm-start (``loss.aux_caption_ce_weight``,
        off by default) and by the causality tests. Position ``t`` predicts ``input_ids[t]``
        from ``[BOS, input_ids[:t]]``.
        """
        memory, memory_mask = self.encode_image(images)
        bos = torch.full(
            (input_ids.shape[0], 1),
            self.bos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        context = torch.cat([bos, input_ids[:, :-1]], dim=1)
        out = self.text_decoder(
            input_ids=context,
            memory=memory,
            memory_mask=memory_mask,
            attention_mask=attention_mask,
        )
        return out.logits

    # ------------------------------------------------------- autoregressive rollout
    def sample_latent(
        self,
        images: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        straight_through: bool = True,
        stochastic: bool = True,
        generator: Optional[torch.Generator] = None,
        latent_length: Optional[int] = None,
    ) -> LatentSample:
        """Draw ``z ~ q_φ(z | x)`` with the differentiable rollout of §6 step 1.

        Parameters
        ----------
        temperature:
            τ of Eq. 3 (typically supplied by
            :class:`~project.models.gumbel.TemperatureScheduler`).
        hard, straight_through:
            Passed to :func:`~project.models.gumbel.gumbel_softmax_sample`.
        stochastic:
            ``False`` disables the Gumbel noise, giving a deterministic argmax rollout
            (used for reconstruction/evaluation, i.e. the MAP latent).
        """
        length = self.latent_length if latent_length is None else int(latent_length)
        if length < 1:
            raise ValueError("latent_length must be >= 1")

        memory, memory_mask = self.encode_image(images)
        batch = images.shape[0]
        device = memory.device

        step_input = self.text_decoder.embed_tokens(
            torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=device)
        )
        cache = None
        logits_steps: List[torch.Tensor] = []
        y_steps: List[torch.Tensor] = []
        y_soft_steps: List[torch.Tensor] = []
        id_steps: List[torch.Tensor] = []
        relaxed_steps: List[torch.Tensor] = []
        hard_steps: List[torch.Tensor] = []
        entropy_steps: List[torch.Tensor] = []

        for _ in range(length):
            out = self.text_decoder(
                inputs_embeds=step_input,
                memory=memory,
                memory_mask=memory_mask,
                cache=cache,
                use_cache=True,
            )
            cache = out.cache
            step_logits = out.logits[:, -1, :].float()  # l_t, Eq. 2

            # ``stochastic=False`` zeroes the Gumbel noise, turning Eq. 3 into a plain
            # tempered softmax (a deterministic argmax rollout) while keeping the same
            # simplex-valued output and gradient path.
            sample = gumbel_softmax_sample(
                step_logits,
                temperature=temperature,
                hard=hard,
                straight_through=straight_through,
                generator=generator if stochastic else None,
                noise=None if stochastic else torch.zeros_like(step_logits),
            )
            y_t, y_soft_t = sample.y, sample.y_soft
            log_probs, ids_t = sample.log_probs, sample.indices

            logits_steps.append(step_logits)
            y_steps.append(y_t)
            y_soft_steps.append(y_soft_t)
            id_steps.append(ids_t)
            # Eq. 2 scored two ways (see plan ambiguity #3).
            relaxed_steps.append((y_t * log_probs).sum(dim=-1))
            hard_steps.append(log_probs.gather(-1, ids_t.unsqueeze(-1)).squeeze(-1))
            entropy_steps.append(-(log_probs.exp() * log_probs).sum(dim=-1))

            # §5.1-style embedding of the sampled (soft) token, fed back as next input.
            step_input = self.text_decoder.embed_soft(y_t).unsqueeze(1)

        return LatentSample(
            logits=torch.stack(logits_steps, dim=1),
            y=torch.stack(y_steps, dim=1),
            y_soft=torch.stack(y_soft_steps, dim=1),
            token_ids=torch.stack(id_steps, dim=1),
            token_log_q_relaxed=torch.stack(relaxed_steps, dim=1),
            token_log_q_hard=torch.stack(hard_steps, dim=1),
            entropy=torch.stack(entropy_steps, dim=1),
            temperature=float(temperature),
        )

    @torch.no_grad()
    def encode_to_tokens(
        self,
        images: torch.Tensor,
        temperature: float = 0.1,
        stochastic: bool = False,
        latent_length: Optional[int] = None,
    ) -> LatentSample:
        """Inference-time image → latent text (no gradients, near-discrete by default)."""
        return self.sample_latent(
            images,
            temperature=temperature,
            hard=True,
            straight_through=False,
            stochastic=stochastic,
            latent_length=latent_length,
        )

    # ------------------------------------------------------------------- utilities
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"latent_length={self.latent_length}, vocab_size={self.vocab_size}"
