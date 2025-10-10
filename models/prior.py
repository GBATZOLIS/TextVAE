import torch
from torch import nn
from tqdm import tqdm

import config


class Prior(nn.Module):
    """
    An LSTM-based autoregressive prior.
    This model's complexity is now configured via arguments to its constructor,
    making it more modular and decoupled from the global config file.
    """

    def __init__(self, num_layers, embedding_dim, dropout):
        super().__init__()

        # The embedding layer converts discrete code indices into continuous vectors.
        self.embedding = nn.Embedding(config.NUM_EMBEDDINGS, embedding_dim)

        # The LSTM processes the sequence of embeddings. Dropout is included for regularization.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1 else 0
            ),  # Dropout is not applied if num_layers is 1
        )

        # The output layer projects the LSTM's hidden states back to logits
        # for each possible code in the codebook.
        self.output_projection = nn.Linear(embedding_dim, config.NUM_EMBEDDINGS)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A sequence of latent code indices, shape (B, N).
        """
        # Embed the input sequence
        x = self.embedding(x)
        # Process with the LSTM
        x, _ = self.lstm(x)
        # Project to logits
        logits = self.output_projection(x)
        return logits

    @torch.no_grad()
    def generate(self, n_samples, device):
        """Autoregressively generates a sequence of latent codes."""
        # Start with a random code for each sample
        generated_codes = torch.randint(
            0, config.NUM_EMBEDDINGS, (n_samples, 1), device=device
        )

        print("Generating code sequences with LSTM prior...")
        # Generate one code at a time
        for _ in tqdm(range(config.NUM_PATCHES - 1)):
            logits = self(generated_codes)
            # We only care about the logits for the very next code
            next_code_logits = logits[:, -1, :]
            # Sample the next code from the probability distribution
            probs = torch.softmax(next_code_logits, dim=-1)
            next_code = torch.multinomial(probs, num_samples=1)
            # Append the new code to our sequence
            generated_codes = torch.cat([generated_codes, next_code], dim=1)

        return generated_codes
