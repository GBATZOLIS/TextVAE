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
            num_embeddings=config.NUM_EMBEDDINGS,
            embedding_dim=config.EMBEDDING_DIM,
            beta=config.BETA,
        )
        self.decoder = ViTDecoder()

    def forward(self, images, seq_len=None):
        """
        Forward pass for the VQ-VAE.
        Args:
            images (torch.Tensor): The input images.
            seq_len (int, optional): If provided, the sequence of latent codes
                                     will be truncated to this length. Defaults to None.
        """
        # Encode the input image and quantize the features
        patches = self.patch_embedding(images)
        encoded_features = self.encoder(patches)
        quantized_features, vq_loss, perplexity = self.quantizer(encoded_features)

        # --- NEW: Apply truncation based on the seq_len argument ---
        if seq_len is not None:
            b, n, d = quantized_features.shape
            # Create a mask to zero out tokens after the specified seq_len
            mask = torch.arange(n, device=images.device)[None, :] < seq_len
            quantized_features = quantized_features * mask.unsqueeze(-1)

        # Decode the (potentially truncated) features to reconstruct the image
        reconstructed_images = self.decoder(quantized_features)

        return {
            "reconstructions": reconstructed_images,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }
