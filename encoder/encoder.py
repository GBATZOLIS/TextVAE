import torch
import torch.nn as nn
from .encoder_config import EncoderConfig


class CountdownPositionalEmbedding(nn.Module):
    """
    Encodes the distance to the Goal.
    Instead of Pos 0, 1, 2... we encode (Target-0), (Target-1)...
    When the embedding represents '0', the model knows it is AT the finish line.
    """

    def __init__(self, dim, max_len=100):
        super().__init__()
        # 0 to max_len. 0 is the "Finish Line".
        self.embedding = nn.Embedding(max_len + 1, dim)
        self.max_len = max_len

    def forward(self, seq_len, target_lengths, device):
        """
        seq_len: Current sequence length (T)
        target_lengths: Tensor of shape [B] containing the Goal Index
        """
        B = target_lengths.shape[0]

        # Current Positions: [0, 1, 2, ... T-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        # Target: [Goal, Goal, ...]
        targets = target_lengths.unsqueeze(1).expand(B, seq_len)

        # Distance: [Goal, Goal-1, ..., Goal-(T-1)]
        # Example: Goal=3. Seq=[0,1,2]. Dist=[3, 2, 1].
        # Wait, usually EOS is at index (Length-1).
        # If Goal=3 (Length=3), indices are 0,1,2.
        # At index 2 (EOS), we want Distance to be 0.
        # So Formula: (Goal - 1) - Position

        # Let's verify: Length=3. Indices=[0,1,2]. EOS is at 2.
        # target_lengths passed from dataset is 3.
        # (3-1) - 0 = 2
        # (3-1) - 1 = 1
        # (3-1) - 2 = 0 (At EOS position, distance is 0)

        dist = (targets - 1) - positions
        dist = torch.clamp(dist, min=0, max=self.max_len)

        return self.embedding(dist)


class VisionEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=512, depth=6, heads=8):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        return self.blocks(x)


class PlanningAutoencoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # 1. Vision
        self.encoder = VisionEncoder(
            config.img_size,
            config.patch_size,
            config.vit_dim,
            config.vit_depth,
            config.heads,
        )

        # 2. Text Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.vit_dim)

        # 3. The "Attention to Position" Mechanism
        # This replaces standard positional encoding
        self.countdown_pos = CountdownPositionalEmbedding(
            config.vit_dim, config.max_len
        )

        # 4. Decoder
        layer = nn.TransformerDecoderLayer(
            d_model=config.vit_dim,
            nhead=config.heads,
            dim_feedforward=config.vit_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.vit_depth)

        self.head = nn.Linear(config.vit_dim, config.vocab_size)

    def forward(self, images, input_ids, target_lengths):
        """
        target_lengths: [B] - The exact integer position where EOS should appear + 1.
        """
        B, L = input_ids.shape

        memory = self.encoder(images)

        # Embed Tokens
        x = self.token_emb(input_ids)

        # Add Countdown Information
        # This injects the "Distance to EOS" into every token's vector.
        # Attention naturally compares these vectors.
        # When a query with 'Dist=1' looks at keys, it knows it needs to wrap up.
        countdown = self.countdown_pos(L, target_lengths, input_ids.device)
        x = x + countdown

        # Standard Causal Mask
        tgt_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=input_ids.device), diagonal=1
        )

        out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        return self.head(out)
