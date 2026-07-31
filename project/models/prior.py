"""§4 — the frozen autoregressive LM prior ``p(z) = Π_t p_LM(z_t | z_<t)``, plus the
soft-token embedding ``z'_t = E_LM^T y_t`` of §5.1.

Every parameter of the language model is frozen and never receives an update; the prior
contributes to the objective solely through the KL term of Eq. 8. Gradients *do* flow
back through the prior's activations to the encoder when the relaxed KL estimator is used
(the LM acts as a fixed differentiable scoring function of ``y``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn

from ..utils.config import PriorConfig

if TYPE_CHECKING:  # annotation-only import (transformers is loaded lazily below)
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

__all__ = ["FrozenLanguageModelPrior", "PriorLogProb"]

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class PriorLogProb:
    """``log p(z)`` for a batch of latent sequences.

    Attributes
    ----------
    total:
        ``Σ_t log p_LM(z_t | z_<t)``, shape ``(B,)``.
    per_token:
        The individual ``log p_LM(z_t | z_<t)`` terms, shape ``(B, T)``.
    logits:
        Prior logits used for scoring, shape ``(B, T, V)`` (position ``t`` predicts
        token ``t``).
    """

    total: torch.Tensor
    per_token: torch.Tensor
    logits: torch.Tensor


class FrozenLanguageModelPrior(nn.Module):
    """A pretrained, frozen causal LM used as the latent prior.

    Parameters
    ----------
    config:
        :class:`~project.utils.config.PriorConfig`. ``pretrained=False`` builds a small
        randomly initialised GPT-2 from ``overrides`` — used by the unit tests so they
        stay offline; production runs load real weights (GPT-2, GPT-Neo, Llama, ...).
    """

    def __init__(self, config: PriorConfig) -> None:
        super().__init__()
        self.config = config
        self.model: "PreTrainedModel"
        self.model, self.tokenizer = self._build(config)

        # --- §4.1: freeze everything, permanently. -----------------------------------
        self.model.requires_grad_(False)
        self.model.eval()

        embedding_shape = self.embedding_matrix.shape
        self._vocab_size = int(embedding_shape[0])
        self._hidden_size = int(embedding_shape[1])
        self._bos_token_id = self._resolve_bos_token_id()

    # ---------------------------------------------------------------- construction
    @staticmethod
    def _build(config: PriorConfig) -> Tuple["PreTrainedModel", Optional[Any]]:
        dtype = _DTYPES.get(config.dtype.lower())
        if dtype is None:
            raise ValueError(f"unknown prior dtype {config.dtype!r}")

        if config.pretrained:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "Loading frozen LM prior %s (dtype=%s)", config.name, config.dtype
            )
            model = AutoModelForCausalLM.from_pretrained(config.name, dtype=dtype)
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=True)
            except Exception as exc:  # pragma: no cover - tokenizer-less checkpoints
                logger.warning(
                    "No tokenizer for %s (%s); latent text decoding disabled.",
                    config.name,
                    exc,
                )
                tokenizer = None
            if tokenizer is not None and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer

        # Randomly initialised GPT-2 (tests / architecture ablations). Kept offline on
        # purpose: no hub access, no tokenizer.
        from transformers import GPT2Config, GPT2LMHeadModel

        gpt2_config = GPT2Config(**config.overrides)
        random_model: "PreTrainedModel" = GPT2LMHeadModel(gpt2_config)
        # transformers wraps ``.to()`` in a decorator that confuses the type checker.
        random_model.to(dtype=dtype)
        return random_model, None

    def _resolve_bos_token_id(self) -> int:
        """Choose the sequence-start symbol.

        Documented ambiguity: the paper does not specify a start token, and GPT-2 has no
        dedicated BOS. We use the tokenizer's ``bos_token_id`` when it exists, else the
        ``eos_token_id`` (the GPT-2 convention), else token 0. The same token is used to
        condition ``p(z_1)`` and to start the encoder's rollout, so ``q`` and ``p`` are
        scored against identical context.
        """
        for attr in ("bos_token_id", "eos_token_id"):
            value = getattr(self.model.config, attr, None)
            if value is None:
                continue
            value = int(value)
            if 0 <= value < self._vocab_size:
                return value
            # Can happen for down-scaled/randomly initialised models whose config keeps
            # the original special-token ids while the vocabulary was shrunk.
            logger.warning(
                "LM config %s=%d is outside the vocabulary (size %d); falling back to token 0.",
                attr,
                value,
                self._vocab_size,
            )
        return 0

    # ------------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        """``|V|`` — the latent vocabulary."""
        return self._vocab_size

    @property
    def hidden_size(self) -> int:
        """``d_model`` of the LM (the width of ``Z'``)."""
        return self._hidden_size

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def _embedding_table(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """``E_LM ∈ R^{|V|×d_model}`` — the frozen token embedding matrix (§5.1)."""
        return cast(torch.Tensor, self._embedding_table.weight)

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding_matrix.dtype

    @property
    def device(self) -> torch.device:
        return self.embedding_matrix.device

    def train(self, mode: bool = True) -> "FrozenLanguageModelPrior":
        """The prior is always in eval mode (no dropout, no BN updates), frozen."""
        super().train(mode)
        self.model.eval()
        return self

    # -------------------------------------------------------------------- embedding
    def embed_soft(self, y: torch.Tensor) -> torch.Tensor:
        """§5.1 — ``z'_t = E_LM^T y_t`` for a batch of soft token vectors.

        ``y`` has shape ``(B, T, |V|)`` and lies on the simplex; the result has shape
        ``(B, T, d_model)``. For a one-hot ``y`` this is exactly a table lookup, so the
        soft and hard paths coincide in the ``τ → 0`` limit.
        """
        embedding = self.embedding_matrix
        if y.shape[-1] != embedding.shape[0]:
            raise ValueError(
                f"soft token dimension {y.shape[-1]} != prior vocabulary {embedding.shape[0]}"
            )
        return y.to(embedding.dtype) @ embedding

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed discrete token ids with the frozen ``E_LM``."""
        return cast(torch.Tensor, self._embedding_table(ids))

    def _bos_embedding(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        bos = torch.full(
            (batch_size, 1), self._bos_token_id, dtype=torch.long, device=self.device
        )
        return self.embed_ids(bos).to(dtype)

    # ------------------------------------------------------------------ log p(z)
    def log_prob_from_ids(self, token_ids: torch.Tensor) -> PriorLogProb:
        """``log p(z)`` for discrete latents, in a single teacher-forced pass.

        No gradient is required here: with hard token ids the prior term of the KL is
        piecewise constant in ``φ``, so we evaluate it under ``no_grad``.
        """
        if token_ids.dim() != 2:
            raise ValueError(
                f"expected (B, T) token ids, got shape {tuple(token_ids.shape)}"
            )
        token_ids = token_ids.to(self.device)
        batch = token_ids.shape[0]
        bos = torch.full(
            (batch, 1), self._bos_token_id, dtype=torch.long, device=self.device
        )
        # Context = [BOS, z_1, ..., z_{T-1}] so position t predicts z_t.
        context = torch.cat([bos, token_ids[:, :-1]], dim=1)
        with torch.no_grad():
            logits = self.model(input_ids=context).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return PriorLogProb(
            total=per_token.sum(dim=-1), per_token=per_token, logits=logits
        )

    def log_prob_from_soft(self, y: torch.Tensor) -> PriorLogProb:
        """``log p(z)`` for a relaxed sample, differentiable w.r.t. ``y``.

        The prior is conditioned on the soft embeddings ``E^T y`` of the previous tokens
        and each term is the relaxed expectation ``Σ_k y_{t,k} log p_LM(k | z_<t)``. With
        a one-hot ``y`` this reduces exactly to :meth:`log_prob_from_ids`.
        """
        if y.dim() != 3:
            raise ValueError(
                f"expected (B, T, V) soft tokens, got shape {tuple(y.shape)}"
            )
        y = y.to(self.device)
        embeds = self.embed_soft(y)
        bos = self._bos_embedding(y.shape[0], embeds.dtype)
        context = torch.cat([bos, embeds[:, :-1]], dim=1)
        logits = self.model(inputs_embeds=context).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = (y.float() * log_probs).sum(dim=-1)
        return PriorLogProb(
            total=per_token.sum(dim=-1), per_token=per_token, logits=logits
        )

    def log_prob(
        self,
        y: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        estimator: str = "relaxed",
    ) -> PriorLogProb:
        """Dispatch to the relaxed or hard estimator (see ``loss.kl_estimator``)."""
        if estimator == "relaxed":
            if y is None:
                raise ValueError("the relaxed estimator requires soft tokens y")
            return self.log_prob_from_soft(y)
        if estimator == "hard":
            if token_ids is None:
                raise ValueError("the hard estimator requires token ids")
            return self.log_prob_from_ids(token_ids)
        raise ValueError(f"unknown KL estimator {estimator!r}")

    # ------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample latent sequences ``z ~ p(z)`` (used for prior-sample generation)."""
        if length < 1:
            raise ValueError("length must be >= 1")
        ids = torch.full(
            (batch_size, 1), self._bos_token_id, dtype=torch.long, device=self.device
        )
        past = None
        out: List[torch.Tensor] = []
        step_input = ids
        for _ in range(length):
            result = self.model(
                input_ids=step_input, past_key_values=past, use_cache=True
            )
            past = result.past_key_values
            logits = result.logits[:, -1, :].float() / max(temperature, 1e-6)
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=generator)
            out.append(next_ids)
            step_input = next_ids
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------------- decoding
    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        """Render latent sequences as human-readable strings (the interpretability payoff)."""
        ids = token_ids.detach().cpu().tolist()
        if self.tokenizer is None:
            return [" ".join(f"<{t}>" for t in row) for row in ids]
        return [
            self.tokenizer.decode(row, skip_special_tokens=skip_special_tokens).strip()
            for row in ids
        ]

    def token_strings(self, token_ids: torch.Tensor) -> List[List[str]]:
        """Per-token strings, for token-level latent inspection.

        Accepts ``(B, T)`` or a single ``(T,)`` sequence (returned as one row).
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        ids = token_ids.detach().cpu().tolist()
        if self.tokenizer is None:
            return [[f"<{t}>" for t in row] for row in ids]
        return [self.tokenizer.convert_ids_to_tokens(row) for row in ids]

    def encode_text(self, texts: Sequence[str], length: int) -> torch.Tensor:
        """Tokenise text into a fixed-length latent (for text → image decoding)."""
        if self.tokenizer is None:
            raise RuntimeError("this prior has no tokenizer; cannot encode text")
        batch = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="pt",
        )
        return batch["input_ids"].to(self.device)

    def one_hot(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One-hot encode ids into the simplex representation ``y`` used by the decoder."""
        return torch.nn.functional.one_hot(
            token_ids.long(), num_classes=self.vocab_size
        ).to(self.embedding_matrix.dtype)

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"name={self.config.name!r}, vocab_size={self.vocab_size}, d_model={self.hidden_size}, frozen=True"


def _filter_logits(
    logits: torch.Tensor, top_k: Optional[int] = None, top_p: Optional[float] = None
) -> torch.Tensor:
    """Standard top-k / nucleus filtering for prior sampling."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(
            -1, sorted_idx, sorted_logits
        )
    return logits
