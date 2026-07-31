"""Dataclass-backed configuration tree with YAML I/O and CLI overrides.

Hydra/OmegaConf are not required (and not installed in the reference environment), so
this module implements the equivalent feature set on top of ``dataclasses`` + ``PyYAML``:

* a fully typed, nested config tree (see :class:`Config`),
* YAML loading with single-inheritance (``_base_: other.yaml``),
* dotted-key command line overrides (``model.encoder.latent_length=48``),
* round-trippable serialisation for checkpoints.

Every knob named in the proposal (latent length, temperature and its schedule, beta and
its schedule, learning rate, batch size, diffusion steps, vocabulary/LM choice, model
sizes, optimizer, scheduler, checkpoint interval) is a field somewhere in this tree.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import yaml

__all__ = [
    "Config",
    "ModelConfig",
    "EncoderConfig",
    "VisionEncoderConfig",
    "TextDecoderConfig",
    "PriorConfig",
    "GumbelConfig",
    "DiffusionDecoderConfig",
    "UNetConfig",
    "LoraConfig",
    "LossConfig",
    "DataConfig",
    "OptimConfig",
    "EMAConfig",
    "TrackerConfig",
    "TrainConfig",
    "load_config",
    "apply_overrides",
    "config_to_dict",
    "config_from_dict",
    "save_config",
]


# --------------------------------------------------------------------------------------
# Model configs
# --------------------------------------------------------------------------------------
@dataclass
class LoraConfig:
    """Optional PEFT LoRA adaptation of a large pretrained sub-network.

    ``target_modules`` is empty by default, meaning "use the right names for whichever
    component this is attached to" — GPT-2 uses ``c_attn``/``c_proj``/``c_fc`` while the
    UNet uses ``to_q``/``to_k``/``to_v``/``to_out.0``. A single shared default would be
    wrong for one of them (PEFT raises "No modules were targeted for adaptation").
    """

    enabled: bool = False
    r: int = 32
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    modules_to_save: List[str] = field(default_factory=list)


@dataclass
class VisionEncoderConfig:
    """§2.1(1) — ``H_img = ViT(x) ∈ R^{N×d_vis}``."""

    name: str = "google/vit-base-patch16-224"
    # ``auto`` picks the loader from ``name``; explicit values: vit | clip | siglip | dinov2
    backend: str = "auto"
    pretrained: bool = True
    freeze_backbone: bool = True
    image_size: int = 224
    # Keep the [CLS]/pooled token in ``H_img`` alongside the patch tokens.
    include_cls_token: bool = False
    # symmetric: inputs are diffusion-space images in [-1, 1] and the encoder applies the
    #            backbone's own normalisation itself (default; one tensor serves both the
    #            encoder input and the decoder's reconstruction target).
    # backbone:  inputs are already normalised for this backbone.
    input_range: str = "symmetric"
    # Applied to the HuggingFace vision config when ``pretrained=False`` (tiny test models).
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDecoderConfig:
    """§2.1(2) — causal self-attention + cross-attention over ``H_img`` + LM head."""

    # gpt2: HuggingFace GPT-2 with ``add_cross_attention`` (shares weights/vocab with the
    #       prior).  scratch: the dependency-light decoder implemented in this repo.
    backend: str = "gpt2"
    pretrained: bool = True
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    tie_lm_head: bool = True
    # ``scratch`` backend geometry (also used as HF config overrides when pretrained=False).
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    ffn_mult: int = 4
    overrides: Dict[str, Any] = field(default_factory=dict)
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class EncoderConfig:
    """§2 — ``q_φ(z | x) = Π_t q_φ(z_t | z_<t, x)``."""

    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    text_decoder: TextDecoderConfig = field(default_factory=TextDecoderConfig)
    # T in ``z = (z_1, ..., z_T)``.
    latent_length: int = 32
    # Projection H_img (d_vis) -> decoder width (d_model).
    projector_layers: int = 2
    projector_dropout: float = 0.1


@dataclass
class PriorConfig:
    """§4.1 — frozen autoregressive LM defining ``p(z) = Π_t p_LM(z_t | z_<t)``."""

    name: str = "gpt2"
    pretrained: bool = True
    dtype: str = "float32"
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GumbelConfig:
    """§3 — Eq. 3 and the temperature annealing schedule."""

    schedule: str = "exponential"  # constant | linear | exponential | cosine
    tau_start: float = 1.0
    tau_end: float = 0.1
    anneal_steps: int = 50_000
    warmup_steps: int = 0
    hard: bool = True
    straight_through: bool = True
    # Temperature used for evaluation/reconstruction (low => near-discrete).
    eval_temperature: float = 0.1


@dataclass
class UNetConfig:
    """``ε_θ`` — Stable-Diffusion-style UNet with cross-attention over ``Z'``."""

    pretrained: bool = False
    name: str = "runwayml/stable-diffusion-v1-5"
    subfolder: str = "unet"
    sample_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    layers_per_block: int = 2
    down_block_types: List[str] = field(
        default_factory=lambda: [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ]
    )
    up_block_types: List[str] = field(
        default_factory=lambda: [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ]
    )
    cross_attention_dim: int = 768
    attention_head_dim: int = 8
    norm_num_groups: int = 32
    gradient_checkpointing: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class DiffusionDecoderConfig:
    """§5 — ``p_θ(x | z)``."""

    # pixel: denoise x_0 directly (faithful to Eq. 7).
    # vae:   latent diffusion through a frozen AutoencoderKL (documented deviation).
    image_space: str = "pixel"
    vae_name: str = "stabilityai/sd-vae-ft-mse"
    vae_subfolder: str = ""
    unet: UNetConfig = field(default_factory=UNetConfig)
    # linear | identity — identity requires cross_attention_dim == d_LM.
    context_projection: str = "linear"
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"  # epsilon | v_prediction
    clip_sample: bool = False
    # Practical extras; both OFF by default so the loss is exactly Eq. 7.
    min_snr_gamma: Optional[float] = None
    cond_dropout_prob: float = 0.0
    # Inference
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    inference_scheduler: str = "ddim"  # ddim | ddpm


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    gumbel: GumbelConfig = field(default_factory=GumbelConfig)
    decoder: DiffusionDecoderConfig = field(default_factory=DiffusionDecoderConfig)


# --------------------------------------------------------------------------------------
# Loss / data / optimisation / training configs
# --------------------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Eq. 8 — ``L = L_diff + β · KL(q ‖ p)``."""

    beta: float = 1.0
    beta_schedule: str = "constant"  # constant | linear | cosine | cyclical
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_steps: int = 0
    beta_anneal_steps: int = 10_000
    beta_cycle_steps: int = 10_000
    # relaxed: score the Gumbel-Softmax sample (differentiable through log p too).
    # hard:    score the discrete argmax ids.
    kl_estimator: str = "relaxed"
    # sum: Eq. 6 sums over t.  mean_per_token: divide by T (practical rescaling).
    kl_reduction: str = "sum"
    # Optional KL floor (free bits, nats per token). 0.0 => exact Eq. 8.
    free_bits: float = 0.0
    # Optional auxiliary teacher-forced caption CE warm-start (not part of the paper).
    aux_caption_ce_weight: float = 0.0


@dataclass
class DataConfig:
    # coco | image_folder | conceptual_captions | laion | hf_captions | synthetic
    dataset: str = "coco"
    root: str = "data/images"
    ann_file: str = "data/annotations/captions_train2017.json"
    # image_folder only: caption sidecar outside ``root``. Accepts a JSON list of
    # {"file_name", "caption"} records, a {file_name: caption} dict, or a .jsonl file.
    # When empty, ``root/captions.json`` and ``root/captions.jsonl`` are auto-detected.
    caption_file: str = ""
    split: str = "train"
    hf_name: str = ""
    hf_config: str = ""
    streaming: bool = False
    image_key: str = "image"
    caption_key: str = "caption"
    image_size: int = 64
    center_crop: bool = True
    random_flip: bool = True
    max_samples: Optional[int] = None
    return_captions: bool = True
    # DataLoader
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2
    drop_last: bool = True
    persistent_workers: bool = False
    shuffle: bool = True


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-4
    # Optional per-group overrides; ``None`` => use ``lr``.
    encoder_lr: Optional[float] = None
    decoder_lr: Optional[float] = None
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 500
    min_lr_ratio: float = 0.0


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    warmup_steps: int = 0
    use_for_eval: bool = True


@dataclass
class TrackerConfig:
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "text-latent-vae"
    wandb_entity: Optional[str] = None
    log_images: bool = True
    log_latent_text: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "runs/textvae"
    run_name: str = "default"
    seed: int = 42
    max_steps: int = 100_000
    max_epochs: Optional[int] = None
    mixed_precision: str = "no"  # no | fp16 | bf16
    log_every: int = 10
    val_every: int = 1_000
    sample_every: int = 1_000
    ckpt_every: int = 1_000
    keep_last_n: int = 3
    val_max_batches: int = 20
    num_log_samples: int = 4
    resume: str = ""  # "" | "auto" | path/to/checkpoint
    allow_tf32: bool = True
    ema: EMAConfig = field(default_factory=EMAConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # When unset, validation reuses ``data`` with ``shuffle=False``.
    val_data: Optional[DataConfig] = None
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ------------------------------------------------------------------
    def validate(self) -> None:
        enc, dec = self.model.encoder, self.model.decoder
        if enc.latent_length < 1:
            raise ValueError("model.encoder.latent_length must be >= 1")
        if enc.text_decoder.backend not in {"gpt2", "scratch"}:
            raise ValueError(
                f"unknown text decoder backend {enc.text_decoder.backend!r}"
            )
        if dec.image_space not in {"pixel", "vae"}:
            raise ValueError(f"unknown decoder.image_space {dec.image_space!r}")
        if dec.context_projection not in {"linear", "identity"}:
            raise ValueError(
                f"unknown decoder.context_projection {dec.context_projection!r}"
            )
        if dec.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError(f"unknown decoder.prediction_type {dec.prediction_type!r}")
        if self.loss.kl_estimator not in {"relaxed", "hard"}:
            raise ValueError(f"unknown loss.kl_estimator {self.loss.kl_estimator!r}")
        if self.loss.kl_reduction not in {"sum", "mean_per_token"}:
            raise ValueError(f"unknown loss.kl_reduction {self.loss.kl_reduction!r}")
        if self.model.gumbel.schedule not in {
            "constant",
            "linear",
            "exponential",
            "cosine",
        }:
            raise ValueError(f"unknown gumbel.schedule {self.model.gumbel.schedule!r}")
        if self.loss.beta_schedule not in {"constant", "linear", "cosine", "cyclical"}:
            raise ValueError(f"unknown loss.beta_schedule {self.loss.beta_schedule!r}")
        if self.train.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"unknown train.mixed_precision {self.train.mixed_precision!r}"
            )
        if self.optim.grad_accum_steps < 1:
            raise ValueError("optim.grad_accum_steps must be >= 1")
        if self.model.gumbel.tau_start <= 0 or self.model.gumbel.tau_end <= 0:
            raise ValueError("Gumbel temperatures must be > 0")

    # -- convenience ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], config_to_dict(self))

    def to_yaml(self) -> str:
        return str(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def validation_data(self) -> DataConfig:
        """Validation data config, defaulting to ``data`` with shuffling disabled."""
        if self.val_data is not None:
            return self.val_data
        val = dataclasses.replace(
            self.data, shuffle=False, drop_last=False, random_flip=False
        )
        return val


# --------------------------------------------------------------------------------------
# (de)serialisation helpers
# --------------------------------------------------------------------------------------
T = TypeVar("T")


def config_to_dict(cfg: Any) -> Any:
    """Recursively convert a (nested) dataclass into plain YAML-safe containers."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {
            f.name: config_to_dict(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)
        }
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _unwrap_optional(tp: Any) -> Any:
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0] if len(args) == 1 else Any


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Coerce a YAML-parsed value into the annotated type."""
    if tp is Any or tp is None:
        return value
    if _is_optional(tp):
        if value is None:
            return None
        tp = _unwrap_optional(tp)

    origin = get_origin(tp)
    if is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise TypeError(
                f"{path}: expected a mapping for {tp.__name__}, got {type(value).__name__}"
            )
        return config_from_dict(tp, value, path=path)
    if origin in (list, List):
        (item_tp,) = get_args(tp) or (Any,)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (tuple, Tuple):
        args = get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)
            )
        return tuple(
            _coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args))
        )
    if origin in (dict, Dict):
        args = get_args(tp) or (Any, Any)
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {
            _coerce(k, args[0], path): _coerce(v, args[1], f"{path}.{k}")
            for k, v in value.items()
        }
    if tp is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if tp is int and not isinstance(value, bool):
        return int(value)
    if tp is float:
        return float(value)
    if tp is str:
        return str(value)
    return value


def config_from_dict(cls: Type[T], data: Optional[Dict[str, Any]], path: str = "") -> T:
    """Build a (nested) dataclass from a plain dict, validating unknown keys."""
    if data is None:
        return cls()
    hints = typing.get_type_hints(cls)
    dataclass_cls = cast(Any, cls)
    known = {f.name for f in dataclasses.fields(dataclass_cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise KeyError(f"unknown config key(s) at {where}: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(dataclass_cls):
        if f.name not in data:
            continue
        sub_path = f"{path}.{f.name}" if path else f.name
        kwargs[f.name] = _coerce(data[f.name], hints[f.name], sub_path)
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_bases(
    path: Union[str, Path], _seen: Optional[set] = None
) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ reference at {path}")
    _seen.add(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping")
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    bases = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = _deep_merge(merged, _load_yaml_with_bases(base_path, _seen))
    return _deep_merge(merged, raw)


def parse_override(text: str) -> Tuple[List[str], Any]:
    """Parse ``a.b.c=value`` into (``["a","b","c"]``, YAML-typed value)."""
    if "=" not in text:
        raise ValueError(f"override {text!r} is not of the form key.path=value")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"override {text!r} has an empty key")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    # YAML 1.1 does not recognise unpunctuated scientific notation ("1e-3" parses as a
    # string), which is exactly how learning rates get typed on the command line.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return key.split("."), value


def apply_overrides(
    data: Dict[str, Any], overrides: Optional[List[str]]
) -> Dict[str, Any]:
    """Apply dotted ``key=value`` overrides to a nested dict (returns a new dict)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    for override in overrides or []:
        keys, value = parse_override(override)
        node: Dict[str, Any] = out
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {} if child is None else child
                if not isinstance(child, dict):
                    raise TypeError(
                        f"cannot descend into non-mapping key {key!r} of {override!r}"
                    )
            else:
                child = dict(child)
            node[key] = child
            node = child
        node[keys[-1]] = value
    return out


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> Config:
    """Load a :class:`Config` from YAML (optional) plus dotted CLI overrides."""
    data: Dict[str, Any] = _load_yaml_with_bases(path) if path is not None else {}
    data = apply_overrides(data, overrides)
    return config_from_dict(Config, data)


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_yaml(), encoding="utf-8")
