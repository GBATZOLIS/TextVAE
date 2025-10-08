# models/decoder.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class DecoderTransformer(nn.Module):
    """
    An autoregressive Transformer that generates an image pixel-by-pixel,
    conditioned on a variable-length sequence of latent vectors.
    """

    def __init__(
        self, n_pixels_rgb, d_model, n_head, n_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.pixel_embedding = nn.Embedding(n_pixels_rgb, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.output_head = nn.Linear(d_model, n_pixels_rgb)
        self.register_buffer(
            "causal_mask", self.generate_square_subsequent_mask(max_seq_len)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, memory, pixels_input):
        pixel_embed = self.pixel_embedding(pixels_input)
        pixel_embed = self.pos_encoder(pixel_embed)

        # The latent sequence (memory) also needs positional encoding
        memory = self.pos_encoder(memory)

        tgt_len = pixels_input.size(1)
        device = pixels_input.device
        mask = self.causal_mask[:tgt_len, :tgt_len].to(device)

        output = self.transformer_decoder(tgt=pixel_embed, memory=memory, tgt_mask=mask)
        logits = self.output_head(output)
        return logits
