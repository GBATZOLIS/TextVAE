import torch
import torch.nn as nn


# --- 1. Core Planning Mechanism ---
class LDPE(nn.Module):
    """
    Length-Difference Positional Encoding.
    Encodes: (Target_Length - Current_Position)
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.embedding = nn.Embedding(max_len + 1, d_model)
        self.max_len = max_len

    def forward(self, x, target_lengths):
        B, Seq, _ = x.shape
        # Positions: [0, 1, 2...]
        positions = torch.arange(Seq, device=x.device).unsqueeze(0)
        # Targets: [Target, Target...]
        targets = target_lengths.unsqueeze(1)

        # Countdown: [Target, Target-1, Target-2...]
        remaining = targets - positions
        remaining = torch.clamp(remaining, min=0, max=self.max_len)

        return self.embedding(remaining)


# --- 2. Vision Transformer (ViT) from Scratch ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)  # [B, Embed, H', W']
        x = x.flatten(2)  # [B, Embed, N_Patches]
        x = x.transpose(1, 2)  # [B, N_Patches, Embed]
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=512, depth=12, heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # Learnable class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embedding
        x = x + self.pos_embed

        # Pass through Transformer
        x = self.blocks(x)
        x = self.norm(x)
        return x


# --- 3. Unified Autoencoder ---
class Encoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        vit_dim=512,
        vit_depth=8,
        vocab_size=50304,
        max_len=100,
    ):
        super().__init__()

        # A. Independent Vision Encoder
        self.vision_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=vit_dim, depth=vit_depth
        )

        # B. Text Decoder
        self.token_emb = nn.Embedding(vocab_size, vit_dim)
        self.ldpe = LDPE(vit_dim, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vit_dim,
            nhead=8,
            dim_feedforward=vit_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.head = nn.Linear(vit_dim, vocab_size)

    def forward(self, images, input_ids, target_lengths):
        """
        images: [B, C, H, W]
        input_ids: [B, Seq_Len] (Tokens so far)
        target_lengths: [B] (The plan)
        """
        # 1. Encode Image -> Visual Memory
        visual_memory = self.vision_encoder(images)  # [B, N_Patches+1, Dim]

        # 2. Embed Text
        tgt_emb = self.token_emb(input_ids)

        # 3. Apply LDPE (The Planning Logic)
        plan_emb = self.ldpe(tgt_emb, target_lengths)
        tgt = tgt_emb + plan_emb

        # 4. Causal Mask
        seq_len = tgt.shape[1]
        tgt_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tgt.device), diagonal=1
        )

        # 5. Decode
        out = self.text_decoder(tgt=tgt, memory=visual_memory, tgt_mask=tgt_mask)

        logits = self.head(out)
        return logits
