# models/vqvae.py

import torch
import torch.nn as nn
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.autoregressive import ElaborationTransformer
from models.decoder import DecoderTransformer


class VQVAE_AR(nn.Module):
    """
    The complete end-to-end model that combines all parts.
    """

    def __init__(self, vq_vae_config, ar_config, decoder_config):
        super().__init__()

        # --- Part 1: VQ Encoder ---
        self.encoder = Encoder(
            in_channels=vq_vae_config["in_channels"],
            num_hiddens=vq_vae_config["num_hiddens"],
            num_residual_layers=vq_vae_config["num_residual_layers"],
            residual_hidden_dim=vq_vae_config["residual_hidden_dim"],
        )
        self.pre_vq_conv = nn.Conv2d(
            vq_vae_config["num_hiddens"], vq_vae_config["embedding_dim"], kernel_size=1
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=vq_vae_config["num_embeddings"],
            embedding_dim=vq_vae_config["embedding_dim"],
            commitment_cost=vq_vae_config["commitment_cost"],
        )

        # --- Part 2: Elaboration Transformer ---
        self.elaboration_transformer = ElaborationTransformer(
            n_codes=ar_config["n_codes"],
            d_model=ar_config["d_model"],
            n_head=ar_config["n_head"],
            n_layers=ar_config["n_layers"],
            max_fixed_len=ar_config["max_fixed_len"],
        )

        # --- Part 3: Pixel Decoder Transformer ---
        self.decoder = DecoderTransformer(
            n_pixels_rgb=decoder_config["n_pixels_rgb"],
            d_model=decoder_config["d_model"],
            n_head=decoder_config["n_head"],
            n_layers=decoder_config["n_layers"],
            max_seq_len=decoder_config["max_seq_len"],
        )

    def forward(self, image, pixels_input, n_generate):
        # --- Stage 1: Get Fixed 64 Tokens ---
        z = self.encoder(image)
        z = self.pre_vq_conv(z)
        vq_output = self.quantizer(z)

        quantized_vectors = vq_output["quantized"].permute(0, 2, 3, 1)
        batch_size, h, w, d_model = quantized_vectors.shape
        quantized_vectors_flat = quantized_vectors.view(batch_size, h * w, d_model)
        fixed_codes = vq_output["encoding_indices"].view(batch_size, h * w)

        # --- Stage 2: Generate N Additional Vectors ---
        generated_vectors = self.elaboration_transformer(fixed_codes, n_generate)

        # --- Stage 3: Combine and Decode ---
        full_latent_sequence = torch.cat(
            [quantized_vectors_flat, generated_vectors], dim=1
        )

        logits = self.decoder(full_latent_sequence, pixels_input)

        return {
            "pixel_logits": logits,
            "vq_loss": vq_output["loss"],
            "perplexity": vq_output["perplexity"],
        }
