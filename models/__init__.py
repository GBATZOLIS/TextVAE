"""Expose high‑level classes so that user code can simply do `from models import TextVAE`"""

from .gumbel_softmax import GumbelSoftmax as GumbelSoftmax
from .text_decoder import TextDecoder as TextDecoder
from .vae import TextVAE as TextVAE  # noqa: F401
from .vae import build_vae_from_config as build_vae_from_config
from .vision_encoder import VisionEncoder as VisionEncoder
