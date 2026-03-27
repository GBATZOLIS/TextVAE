import torch
import torch.nn as nn
from .encoder_config import EncoderConfig


class CountdownEmbedding(nn.Module):
    """
    Injects the 'Remaining Budget' into the model.
    If we want to generate 5 tokens, the inputs will have embeddings for:
    [5, 4, 3, 2, 1]
    When the countdown hits 1 (or 0), the model learns this implies EOS.
    """

    def __init__(self, dim, max_len=256):
        super().__init__()
        # +1 for safety (padding or 0 case)
        self.embedding = nn.Embedding(max_len + 1, dim)
        self.max_len = max_len

    def forward(self, current_positions, target_lengths):
        """
        current_positions: [B, L] (0, 1, 2...)
        target_lengths: [B, 1] or [B] (The Goal, e.g., 5)
        """
        if target_lengths.dim() == 1:
            target_lengths = target_lengths.unsqueeze(1)  # [B, 1]

        # Broadcasting: [B, 1] - [B, L] -> [B, L]
        # Example: Goal=5. Pos=[0, 1, 2].
        # Dist = 5 - 0 = 5
        # Dist = 5 - 1 = 4
        # ...
        distance = target_lengths - current_positions

        # Clamp to ensure we don't go negative (though valid planning shouldn't)
        # We also clamp to max_len to prevent index errors
        distance = torch.clamp(distance, min=0, max=self.max_len)

        return self.embedding(distance)


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

        # 1. Vision Encoder
        self.encoder = VisionEncoder(
            config.img_size,
            config.patch_size,
            config.vit_dim,
            config.vit_depth,
            config.heads,
        )

        # 2. Text Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.vit_dim)
        self.pos_emb = nn.Embedding(config.max_len + 5, config.vit_dim)

        # 3. The Dense Planning Signal
        self.countdown_emb = CountdownEmbedding(
            config.vit_dim, max_len=config.max_len + 5
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

        self.eos_token_id = 50256

    def forward(self, images, input_ids, target_lengths):
        """
        images: [B, C, H, W]
        input_ids: [B, L]
        target_lengths: [B] (The integer goal length)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # --- Vision ---
        memory = self.encoder(images)

        # --- Embeddings ---
        # 1. Word Meaning
        x = self.token_emb(input_ids)

        # 2. Absolute Position (Where am I in the sequence?)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_vec = self.pos_emb(positions)

        # 3. Relative Countdown (How far to the finish line?)
        # This is the heavy lifter for planning.
        countdown_vec = self.countdown_emb(positions, target_lengths)

        # Combine all signals
        x = x + pos_vec + countdown_vec

        # --- Standard Causal Masking ---
        # We don't need the custom Goal mask anymore because the goal info
        # is baked into the input vectors themselves.
        tgt_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        # --- Decode ---
        out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        return self.head(out)
