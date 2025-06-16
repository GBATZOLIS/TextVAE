"""Expose high‑level classes so that user code can simply do `from models import TextVAE`"""
from .vae import TextVAE, build_vae_from_config  # noqa: F401
