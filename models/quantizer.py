import torch
from torch import nn
from sklearn.cluster import KMeans


class VectorQuantizer(nn.Module):
    """
    The Vector Quantizer (VQ) layer for the VQ-VAE.

    This module takes the continuous output of the encoder and maps it to a
    discrete set of learned embedding vectors (the codebook).
    """

    def __init__(self, num_embeddings, embedding_dim, beta):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # The codebook of learned embedding vectors
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize weights with a uniform distribution
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

        # --- NEW: Flag to track if the codebook has been initialized ---
        self.register_buffer("_initialized", torch.tensor(False))

    def forward(self, x):
        """
        Forward pass for the VectorQuantizer.
        Args:
            x (torch.Tensor): The continuous output from the encoder.
                              Shape: (B, num_patches, embedding_dim).
        Returns:
            tuple: A tuple containing:
                   - quantized_features (torch.Tensor): The quantized vectors.
                   - vq_loss (torch.Tensor): The total VQ loss.
                   - perplexity (torch.Tensor): A measure of codebook usage.
        """
        # Reshape input to (B * num_patches, embedding_dim)
        flat_input = x.view(-1, self.embedding_dim)

        # Calculate distances between input vectors and embedding vectors
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Find the closest embedding vector for each input vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Convert indices to one-hot encoding
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the input by looking up the embeddings
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        # --- VQ Loss Calculation ---
        # 1. Codebook Loss: Encourages embeddings to match encoder outputs
        codebook_loss = torch.mean((quantized.detach() - x) ** 2)
        # 2. Commitment Loss: Encourages encoder outputs to commit to an embedding
        commitment_loss = torch.mean((quantized - x.detach()) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Use straight-through estimator to allow gradients to flow
        quantized = x + (quantized - x).detach()

        # --- Perplexity Calculation ---
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity

    @torch.no_grad()
    def k_means_init(self, features):
        """
        Initializes the codebook embeddings using the K-Means algorithm.
        Args:
            features (torch.Tensor): A batch of features from the encoder.
        """
        if self._initialized:
            return

        print("Performing one-time K-Means initialization of the codebook...")
        # Flatten the features and convert to numpy for KMeans
        features_np = features.reshape(-1, self.embedding_dim).cpu().numpy()

        # Run KMeans to find the initial centroids
        kmeans = KMeans(n_clusters=self.num_embeddings, n_init="auto").fit(features_np)

        # Assign the learned centroids to the embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))

        # Mark the codebook as initialized
        self._initialized.fill_(True)
