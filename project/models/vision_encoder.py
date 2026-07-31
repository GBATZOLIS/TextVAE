"""§2.1(1) — the vision encoder producing ``H_img = ViT(x) ∈ R^{N×d_vis}``.

Supported backbones: ViT-B/16 (default), CLIP ViT, SigLIP, DINOv2. Any HuggingFace
checkpoint of those families works; ``pretrained=False`` builds a randomly initialised
model from the same architecture (used by the unit tests, which must stay offline).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..datasets.transforms import normalize_for_backbone
from ..utils.config import VisionEncoderConfig

logger = logging.getLogger(__name__)

__all__ = ["VisionEncoder", "VisionEncoderOutput"]


@dataclass
class VisionEncoderOutput:
    """``H_img`` plus the mask the text decoder should use when cross-attending."""

    hidden_states: torch.Tensor  # (B, N, d_vis)
    attention_mask: torch.Tensor  # (B, N), all ones for fixed-size images


def _resolve_backend(config: VisionEncoderConfig) -> str:
    if config.backend != "auto":
        return config.backend
    name = config.name.lower()
    if "siglip" in name:
        return "siglip"
    if "clip" in name:
        return "clip"
    if "dinov2" in name:
        return "dinov2"
    return "vit"


class VisionEncoder(nn.Module):
    """Wraps a HuggingFace vision transformer into a uniform ``x -> H_img`` interface."""

    # Backbones whose first output token is a class/pooling token rather than a patch.
    _CLS_BACKENDS = {"vit", "clip", "dinov2"}

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backend = _resolve_backend(config)
        self.model = self._build()
        self._hidden_size = int(self._model_config().hidden_size)
        self._supports_interpolation = (
            "interpolate_pos_encoding"
            in inspect.signature(self.model.forward).parameters
        )

        if config.freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()
            logger.info(
                "Vision backbone %s frozen (%d params).",
                config.name,
                self.num_parameters,
            )

    # ---------------------------------------------------------------- construction
    def _build(self) -> nn.Module:
        from transformers import (
            CLIPVisionConfig,
            CLIPVisionModel,
            Dinov2Config,
            Dinov2Model,
            SiglipVisionConfig,
            SiglipVisionModel,
            ViTConfig,
            ViTModel,
        )

        model_cls, config_cls = {
            "vit": (ViTModel, ViTConfig),
            "clip": (CLIPVisionModel, CLIPVisionConfig),
            "siglip": (SiglipVisionModel, SiglipVisionConfig),
            "dinov2": (Dinov2Model, Dinov2Config),
        }[self.backend]

        # We only ever read patch tokens, so drop the pooling head where the class
        # supports it: an unused pooler would otherwise sit in the parameter list
        # collecting no gradients.
        kwargs: Dict[str, Any] = {}
        if "add_pooling_layer" in inspect.signature(model_cls).parameters:
            kwargs["add_pooling_layer"] = False

        if self.config.pretrained:
            logger.info(
                "Loading pretrained vision backbone %s (%s).",
                self.config.name,
                self.backend,
            )
            return model_cls.from_pretrained(self.config.name, **kwargs)

        overrides = dict(self.config.overrides)
        overrides.setdefault("image_size", self.config.image_size)
        return model_cls(config_cls(**overrides), **kwargs)

    def _model_config(self) -> Any:
        cfg = self.model.config
        # CLIP/SigLIP wrap the vision settings in a sub-config when the full model is used.
        return getattr(cfg, "vision_config", cfg)

    # ------------------------------------------------------------------ properties
    @property
    def hidden_size(self) -> int:
        """``d_vis`` — width of ``H_img``."""
        return self._hidden_size

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def patch_size(self) -> int:
        return int(getattr(self._model_config(), "patch_size", 16))

    def num_patches(self, image_size: Optional[int] = None) -> int:
        """Number of patch tokens ``N`` for a square input of ``image_size``."""
        size = image_size or self.config.image_size
        grid = (size // self.patch_size) ** 2
        return grid + (
            1 if self._has_cls_token and self.config.include_cls_token else 0
        )

    @property
    def _has_cls_token(self) -> bool:
        return self.backend in self._CLS_BACKENDS

    def train(self, mode: bool = True) -> "VisionEncoder":
        """A frozen backbone stays in eval mode (no dropout / stochastic depth)."""
        super().train(mode)
        if self.config.freeze_backbone:
            self.model.eval()
        return self

    # --------------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        """Compute ``H_img = ViT(x)``.

        ``pixel_values``: (B, 3, H, W), normalised as the backbone expects.
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"expected (B, C, H, W) images, got {tuple(pixel_values.shape)}"
            )

        kwargs: Dict[str, Any] = {}
        configured = int(
            getattr(self._model_config(), "image_size", pixel_values.shape[-1])
        )
        if self._supports_interpolation and pixel_values.shape[-1] != configured:
            # Lets a pretrained backbone run at a different resolution than it was
            # trained at by interpolating its position embeddings.
            kwargs["interpolate_pos_encoding"] = True

        if self.config.input_range == "symmetric":
            # Datasets hand us the diffusion-space image in [-1, 1]; convert it to this
            # backbone's expected normalisation here so callers keep a single tensor.
            pixel_values = normalize_for_backbone(pixel_values, self.backend)
        elif self.config.input_range != "backbone":
            raise ValueError(f"unknown vision input_range {self.config.input_range!r}")

        context = torch.no_grad() if self.config.freeze_backbone else _null_context()
        with context:
            outputs = self.model(
                pixel_values=pixel_values.to(self._param_dtype()), **kwargs
            )
        hidden = outputs.last_hidden_state

        if self._has_cls_token and not self.config.include_cls_token:
            hidden = hidden[:, 1:, :]
        if self.config.freeze_backbone:
            # Detach explicitly: the backbone contributes no gradients, but downstream
            # modules must still build a graph on their own parameters.
            hidden = hidden.detach()

        mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        return VisionEncoderOutput(hidden_states=hidden, attention_mask=mask)

    def _param_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"name={self.config.name!r}, backend={self.backend!r}, d_vis={self.hidden_size}, "
            f"frozen={self.config.freeze_backbone}"
        )


class _null_context:
    """``contextlib.nullcontext`` equivalent kept local for clarity in the forward pass."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def build_vision_encoder(config: VisionEncoderConfig) -> VisionEncoder:
    return VisionEncoder(config)
