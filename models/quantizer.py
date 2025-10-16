# models/quantizer.py

import torch
from torch import nn, Tensor
from sklearn.cluster import KMeans


class VectorQuantizer(nn.Module):
    """
    The Vector Quantizer (VQ) layer with Exponential Moving Average (EMA) updates
    to prevent codebook collapse.
    """

    _ema_cluster_size: Tensor
    _ema_w: Tensor
    _initialized: Tensor

    def __init__(self, num_embeddings, embedding_dim, beta, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", self.embedding.weight.data.clone())

    def forward(self, x):
        flat_input = x.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (
                1 - self.decay
            ) * torch.sum(encodings, 0)

            dw = torch.matmul(encodings.t(), flat_input.detach())
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw

            # --- FIX: Add a small epsilon to prevent division by zero ---
            # This is crucial for numerical stability, especially if some codes
            # are not used for a while, causing their cluster size to decay to zero.
            n = torch.sum(self._ema_cluster_size)
            normalized_cluster_size = self._ema_cluster_size * (
                n / (n + 1e-5)
            ) + 1e-5 * (1 / n)

            self.embedding.weight.data.copy_(
                self._ema_w / normalized_cluster_size.unsqueeze(1)
            )

        codebook_loss = torch.mean((quantized.detach() - x) ** 2)
        commitment_loss = torch.mean((quantized - x.detach()) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity, encoding_indices

    @torch.no_grad()
    def k_means_init(self, features):
        if self._initialized:
            print("Codebook already initialized.")
            return

        print("Performing one-time K-Means initialization of the codebook...")
        features_np = features.reshape(-1, self.embedding_dim).cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.num_embeddings, n_init="auto", random_state=0
        ).fit(features_np)
        self.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
        self._initialized.fill_(True)
