# models/autoregressive.py

import torch
import torch.nn as nn
from models.decoder import PositionalEncoding  # Re-use PositionalEncoding


class ElaborationTransformer(nn.Module):
    """
    An autoregressive Transformer that takes a fixed sequence of codes
    and generates N additional continuous embedding vectors to "elaborate" on the content.
    """

    def __init__(self, n_codes, d_model, n_head, n_layers, max_fixed_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.code_embedding = nn.Embedding(n_codes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_fixed_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, fixed_codes, n_generate):
        embedded_codes = self.code_embedding(fixed_codes)
        embedded_codes = self.pos_encoder(embedded_codes)

        context = self.transformer_encoder(embedded_codes)

        current_state = context.mean(dim=1).unsqueeze(1)

        generated_vectors = []
        for _ in range(n_generate):
            next_vector = self.output_head(current_state)
            generated_vectors.append(next_vector)
            current_state = next_vector

        if not generated_vectors:
            return torch.empty(
                (fixed_codes.size(0), 0, self.d_model), device=fixed_codes.device
            )

        return torch.cat(generated_vectors, dim=1)
