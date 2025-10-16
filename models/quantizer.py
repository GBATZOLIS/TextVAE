# models/quantizer.py

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self._decay = decay
        self._epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

        self.register_buffer("_initialized", torch.tensor(False))
        # --- Buffers for EMA updates ---
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.zeros(num_embeddings, self.embedding_dim))

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

        # --- EMA Codebook Update ---
        if self.training and self._initialized:
            # --- THE FIX: Detach flat_input to prevent graph leakage into the buffer ---
            dw = torch.matmul(encodings.t(), flat_input.detach())

            # Use EMA to update the embedding vectors
            self._ema_cluster_size.data.mul_(self._decay).add_(
                torch.sum(encodings, 0), alpha=1 - self._decay
            )
            self._ema_w.data.mul_(self._decay).add_(dw, alpha=1 - self._decay)

            # Laplace smoothing for cluster size
            n = torch.sum(self._ema_cluster_size)
            smoothed_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon)
                * n
            )

            # Update the codebook
            self.embedding.weight.data.copy_(
                self._ema_w / smoothed_cluster_size.unsqueeze(1)
            )

        codebook_loss = F.mse_loss(quantized.detach(), x)
        commitment_loss = F.mse_loss(quantized, x.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity, encoding_indices.squeeze()

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

        # Initialize EMA buffers
        self._ema_w.data.copy_(self.embedding.weight.data)
        self._ema_cluster_size.data.fill_(
            1.0
        )  # Start with a count of 1 for each cluster

        self._initialized.fill_(True)
