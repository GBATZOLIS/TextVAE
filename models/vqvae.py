# models/vqvae.py

import torch
from torch import nn
import config
from .patches import PatchEmbedding
from .encoder import ViTEncoder
from .quantizer import VectorQuantizer
from .decoder import ViTDecoder


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.encoder = ViTEncoder()
        self.quantizer = VectorQuantizer(
            num_embeddings=config.NUM_EMBEDTINGS,
            embedding_dim=config.EMBEDDING_DIM,
            beta=config.BETA,
        )
        self.decoder = ViTDecoder()

    def forward(self, images, seq_len=None):
        """
        Forward pass for the VQ-VAE.
        Args:
            images (torch.Tensor): The input images.
            seq_len (torch.Tensor, optional): A tensor of sequence lengths for each item
                                              in the batch. Shape: (batch_size,).
                                              Defaults to None.
        """
        # Encode the input image and quantize the features
        patches = self.patch_embedding(images)
        encoded_features = self.encoder(patches)
        quantized_features, vq_loss, perplexity, indeces = self.quantizer(
            encoded_features
        )

        mask = None
        if seq_len is not None:
            b, n, d = quantized_features.shape
            # --- MODIFIED: Create a correctly shaped attention mask ---
            # seq_len is now a tensor of shape (b,).
            # We compare a range [0, 1, ..., n-1] with each length in seq_len.
            # torch.arange is broadcast to (b, n)
            # seq_len.unsqueeze(-1) is broadcast to (b, n)
            # The result is a correctly shaped boolean mask (b, n).
            mask = torch.arange(n, device=images.device)[None, :] >= seq_len.unsqueeze(
                -1
            )

        # Decode the features to reconstruct the image, passing the mask
        reconstructed_images = self.decoder(quantized_features, mask=mask)

        return {
            "reconstructions": reconstructed_images,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "indeces": indeces,
        }
