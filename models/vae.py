# models/vae.py

import torch.nn as nn
from .vision_encoder import VisionEncoder
from .diffusion_decoder import DiffusionDecoder


class TextVAE(nn.Module):
    """A wrapper for all trainable components for easier DeepSpeed integration."""

    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            swin_model_name=config.models.vision_encoder_name,
            max_text_length=config.models.vision_encoder_max_text_length,
        )
        self.diffusion_decoder = DiffusionDecoder(
            unet_model_name=config.models.diffusion_decoder_unet_name,
            scheduler_type=config.models.diffusion_decoder_scheduler_type,
            chunk_size=config.models.diffusion_decoder_chunk_size,
        )

    def forward(self, original_images):
        generated_logits, soft_embeddings = self.vision_encoder(original_images)
        soft_embeddings_casted = soft_embeddings.half()
        loss_recon = self.diffusion_decoder(soft_embeddings_casted, original_images)

        return generated_logits, soft_embeddings, loss_recon
