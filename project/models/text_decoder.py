"""§2.1(2) — the autoregressive text decoder that turns ``H_img`` into vocabulary logits.

Each layer performs

1. **causal self-attention** over the previously generated tokens ``z_<t``,
2. **cross-attention** over the image features ``H_img``,
3. a position-wise MLP,

and a final linear "language model head" produces ``l_t ∈ R^{|V|}`` (Eq. 2).

Two interchangeable backends implement that same interface:

``gpt2``
    HuggingFace ``GPT2LMHeadModel`` with ``add_cross_attention=True``. Loading the
    pretrained checkpoint means the encoder starts from the *same* weights and vocabulary
    as the frozen prior of §4 (the cross-attention layers are new and randomly
    initialised, since GPT-2 has none). Optional LoRA adaptation is supported.
``scratch``
    A self-contained implementation in this file — no external checkpoint, explicit KV
    cache, useful for tests, ablations and small-scale runs.

Both support teacher forcing (a full parallel pass over a given token sequence) and
incremental decoding with a KV cache, the latter being what the differentiable
autoregressive Gumbel rollout of §6 needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import TextDecoderConfig

logger = logging.getLogger(__name__)

__all__ = [
    "TextDecoderOutput",
    "TextDecoderBase",
    "GPT2CrossAttentionDecoder",
    "ScratchCrossAttentionDecoder",
    "build_text_decoder",
]


@dataclass
class TextDecoderOutput:
    """Logits ``(B, L, |V|)`` and, optionally, the updated KV cache."""

    logits: torch.Tensor
    cache: Optional[Any] = None
    hidden_states: Optional[torch.Tensor] = None


class TextDecoderBase(nn.Module):
    """Interface shared by both decoder backends."""

    vocab_size: int
    d_model: int

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed_soft(self, y: torch.Tensor) -> torch.Tensor:
        """``E^T y`` using the decoder's *own* input embedding table.

        Note this is the encoder-side embedding (which is trainable and, for the ``gpt2``
        backend, initialised from the LM). The decoder-side conditioning of §5.1 uses the
        *frozen prior's* matrix instead — see
        :meth:`project.models.prior.FrozenLanguageModelPrior.embed_soft`.
        """
        table = self.input_embedding_matrix
        if y.shape[-1] != table.shape[0]:
            raise ValueError(f"soft token dim {y.shape[-1]} != vocab {table.shape[0]}")
        return y.to(table.dtype) @ table

    @property
    def input_embedding_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
        use_cache: bool = False,
    ) -> TextDecoderOutput:
        raise NotImplementedError

    @staticmethod
    def _check_inputs(
        input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]
    ) -> None:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")


# ======================================================================================
# HuggingFace GPT-2 backend
# ======================================================================================
class GPT2CrossAttentionDecoder(TextDecoderBase):
    """GPT-2 with cross-attention injected into every block."""

    def __init__(
        self, config: TextDecoderConfig, vocab_size: int, lm_name: str = "gpt2"
    ) -> None:
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel

        self.config = config
        # ``nn.Module`` because PEFT may wrap the GPT-2 model in a PeftModel below.
        self.model: nn.Module
        if config.pretrained:
            gpt2_config = GPT2Config.from_pretrained(lm_name)
            gpt2_config.add_cross_attention = True
            for key, value in config.overrides.items():
                setattr(gpt2_config, key, value)
            logger.info(
                "Initialising text decoder from %s (+ new cross-attention).", lm_name
            )
            # ``ignore_mismatched_sizes`` is required because the cross-attention weights
            # do not exist in the checkpoint and are randomly initialised here.
            self.model = GPT2LMHeadModel.from_pretrained(
                lm_name, config=gpt2_config, ignore_mismatched_sizes=True
            )
        else:
            overrides: Dict[str, Any] = {
                "vocab_size": vocab_size,
                "n_embd": config.d_model,
                "n_layer": config.num_layers,
                "n_head": config.num_heads,
                "n_positions": config.max_position_embeddings,
                "n_inner": config.ffn_mult * config.d_model,
                "resid_pdrop": config.dropout,
                "embd_pdrop": config.dropout,
                "attn_pdrop": config.dropout,
                "add_cross_attention": True,
            }
            overrides.update(config.overrides)
            self.model = GPT2LMHeadModel(GPT2Config(**overrides))

        model_vocab = int(self.model.config.vocab_size)
        if model_vocab != vocab_size:
            raise ValueError(
                f"text decoder vocabulary ({model_vocab}) must match the prior's "
                f"({vocab_size}); use the same LM for both (see plan ambiguity #5)"
            )
        self.vocab_size = model_vocab
        self.d_model = int(self.model.config.n_embd)
        self.max_positions = int(self.model.config.n_positions)
        if config.tie_lm_head:
            self.model.tie_weights()

        if config.lora.enabled:
            self._apply_lora(config)

    def _apply_lora(self, config: TextDecoderConfig) -> None:
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        spec = config.lora
        # Cross-attention is brand new, so it must remain fully trainable (LoRA on top of
        # random weights would waste the layer).
        modules_to_save = spec.modules_to_save or ["crossattention"]
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.r,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            target_modules=spec.target_modules or ["c_attn", "c_proj", "c_fc"],
            modules_to_save=modules_to_save,
            fan_in_fan_out=True,  # GPT-2 uses Conv1D
        )
        self.model = cast(nn.Module, get_peft_model(cast(Any, self.model), peft_config))
        logger.info("Applied LoRA (r=%d) to the text decoder.", spec.r)

    # ------------------------------------------------------------------- properties
    @property
    def _embedding_table(self) -> nn.Module:
        # ``self.model`` is typed as nn.Module (PEFT may wrap it), so the transformers
        # API surface is only visible through an ``Any`` view.
        return cast(nn.Module, cast(Any, self.model).get_input_embeddings())

    @property
    def input_embedding_matrix(self) -> torch.Tensor:
        return cast(torch.Tensor, self._embedding_table.weight)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._embedding_table(input_ids))

    # ---------------------------------------------------------------------- forward
    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
        use_cache: bool = False,
    ) -> TextDecoderOutput:
        self._check_inputs(input_ids, inputs_embeds)
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=cache,
            use_cache=use_cache,
            output_hidden_states=False,
        )
        return TextDecoderOutput(
            logits=outputs.logits,
            cache=outputs.past_key_values if use_cache else None,
        )

    def gradient_checkpointing_enable(self) -> None:
        cast(Any, self.model).gradient_checkpointing_enable()


# ======================================================================================
# Self-contained backend
# ======================================================================================
@dataclass
class LayerCache:
    """Per-layer KV cache. Cross-attention keys/values only depend on ``H_img``, so they
    are computed once per rollout and reused for every step."""

    self_k: Optional[torch.Tensor] = None
    self_v: Optional[torch.Tensor] = None
    cross_k: Optional[torch.Tensor] = None
    cross_v: Optional[torch.Tensor] = None


@dataclass
class ScratchCache:
    layers: List[LayerCache] = field(default_factory=list)

    @property
    def length(self) -> int:
        if not self.layers or self.layers[0].self_k is None:
            return 0
        return int(self.layers[0].self_k.shape[2])


class _MultiHeadAttention(nn.Module):
    """Multi-head attention with an explicit KV cache (self- or cross-attention)."""

    def __init__(
        self, d_model: int, num_heads: int, dropout: float, is_causal: bool
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        # ``None`` selects the cached-cross-attention path: keys/values come from
        # ``past_k``/``past_v`` and the memory is not re-projected.
        key_value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._split(self.q_proj(query))
        if past_k is not None and past_v is not None and key_value is None:
            k, v = past_k, past_v  # cross-attention: reuse cached projections
        else:
            k = self._split(self.k_proj(key_value))
            v = self._split(self.v_proj(key_value))
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S) additive-style boolean mask for SDPA.
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        # Causal masking is only needed when more than one query attends at once; during
        # incremental decoding a single query legitimately sees the whole cache.
        causal = self.is_causal and q.shape[2] > 1 and attn_mask is None
        if self.is_causal and q.shape[2] > 1 and attn_mask is not None:
            raise ValueError(
                "padded causal self-attention is not supported by this backend"
            )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(out), k, v


class _DecoderBlock(nn.Module):
    """Pre-LN block: causal self-attention → image cross-attention → MLP."""

    def __init__(
        self, d_model: int, num_heads: int, ffn_mult: int, dropout: float
    ) -> None:
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = _MultiHeadAttention(
            d_model, num_heads, dropout, is_causal=True
        )
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = _MultiHeadAttention(
            d_model, num_heads, dropout, is_causal=False
        )
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        cache: Optional[LayerCache],
    ) -> Tuple[torch.Tensor, LayerCache]:
        new_cache = LayerCache()

        normed = self.ln_self(x)
        hidden, k, v = self.self_attn(
            normed,
            normed,
            past_k=cache.self_k if cache else None,
            past_v=cache.self_v if cache else None,
        )
        new_cache.self_k, new_cache.self_v = k, v
        x = x + self.dropout(hidden)

        cached_cross = cache is not None and cache.cross_k is not None
        hidden, ck, cv = self.cross_attn(
            self.ln_cross(x),
            None if cached_cross else memory,
            key_padding_mask=memory_mask,
            past_k=cache.cross_k if cache else None,
            past_v=cache.cross_v if cache else None,
        )
        new_cache.cross_k, new_cache.cross_v = ck, cv
        x = x + self.dropout(hidden)

        return x + self.mlp(self.ln_mlp(x)), new_cache


class ScratchCrossAttentionDecoder(TextDecoderBase):
    """Dependency-free implementation of the §2.1(2) decoder."""

    def __init__(self, config: TextDecoderConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        self.d_model = int(config.d_model)
        self.max_positions = int(config.max_position_embeddings)

        self.wte = nn.Embedding(self.vocab_size, self.d_model)
        self.wpe = nn.Embedding(self.max_positions, self.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            _DecoderBlock(
                self.d_model, config.num_heads, config.ffn_mult, config.dropout
            )
            for _ in range(config.num_layers)
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if config.tie_lm_head:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def input_embedding_matrix(self) -> torch.Tensor:
        return self.wte.weight

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
        use_cache: bool = False,
    ) -> TextDecoderOutput:
        self._check_inputs(input_ids, inputs_embeds)
        if attention_mask is not None and (attention_mask == 0).any():
            # Fixed-length latents never need padding; captions used for the optional
            # teacher-forced warm-start are padded to a constant length instead.
            raise ValueError(
                "the scratch backend does not support padded self-attention"
            )
        x = inputs_embeds if input_ids is None else self.embed_tokens(input_ids)
        assert x is not None  # guaranteed by _check_inputs above
        if memory.shape[-1] != self.d_model:
            raise ValueError(
                f"memory width {memory.shape[-1]} != decoder d_model {self.d_model}; "
                "project H_img before calling the decoder"
            )

        past_len = cache.length if isinstance(cache, ScratchCache) else 0
        seq_len = x.shape[1]
        if past_len + seq_len > self.max_positions:
            raise ValueError(
                f"sequence length {past_len + seq_len} exceeds max_position_embeddings "
                f"({self.max_positions})"
            )
        positions = torch.arange(past_len, past_len + seq_len, device=x.device)
        x = self.drop(x + self.wpe(positions)[None])

        layer_caches: List[Optional[LayerCache]] = (
            list(cache.layers)
            if isinstance(cache, ScratchCache) and cache.layers
            else [None] * len(self.blocks)
        )
        new_layers: List[LayerCache] = []
        for block, layer_cache in zip(self.blocks, layer_caches):
            x, updated = block(x, memory, memory_mask, layer_cache)
            new_layers.append(updated)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return TextDecoderOutput(
            logits=logits,
            cache=ScratchCache(layers=new_layers) if use_cache else None,
            hidden_states=hidden,
        )


def build_text_decoder(
    config: TextDecoderConfig, vocab_size: int, lm_name: str = "gpt2"
) -> TextDecoderBase:
    """Instantiate the configured text-decoder backend."""
    if config.backend == "gpt2":
        return GPT2CrossAttentionDecoder(config, vocab_size=vocab_size, lm_name=lm_name)
    if config.backend == "scratch":
        return ScratchCrossAttentionDecoder(config, vocab_size=vocab_size)
    raise ValueError(f"unknown text decoder backend {config.backend!r}")
