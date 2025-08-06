"""Expose high‑level classes so that user code can simply do `from models import TextVAE`"""

from .vae import TextVAE as TextVAE  # noqa: F401
from .vision_encoder import VLMEncoder as VLMEncoder
from .diffusion_decoder import DiffusionDecoder as DiffusionDecoder
