import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torchvision.models as models
from .encoder_config import EncoderConfig


class CountdownEmbedding(nn.Module):
    def __init__(self, dim, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(max_len + 1, dim)
        self.max_len = max_len

    def forward(self, current_positions, target_lengths):
        if target_lengths.dim() == 1:
            target_lengths = target_lengths.unsqueeze(1)

        distance = target_lengths - current_positions
        distance = torch.clamp(distance, min=0, max=self.max_len)
        return self.embedding(distance)


class ResNetEncoder(nn.Module):
    """
    Uses a pre-trained ResNet to extract visual features.
    We strip the classification head and project to GPT-2 dimension.
    """

    def __init__(self, embed_dim=768):  # GPT-2 small is 768
        super().__init__()
        # Load standard ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final FC layer and the pooling layer to keep spatial grid
        # ResNet50 typically outputs [B, 2048, 7, 7] before pooling
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Project 2048 -> GPT2 Dim (768)
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        features = self.backbone(x)  # [B, 2048, 7, 7]
        features = self.proj(features)  # [B, 768, 7, 7]

        # Flatten to sequence: [B, 768, 49] -> [B, 49, 768]
        features = features.flatten(2).transpose(1, 2)
        return features


class PlanningGPT2(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # 1. Vision: Pre-trained ResNet
        # GPT-2 Small has d_model=768
        self.vision_encoder = ResNetEncoder(embed_dim=768)

        # 2. Language: Pre-trained GPT-2
        # We load the base model.
        # Note: You might want to freeze the first few layers if dataset is small.
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

        # 3. The Countdown Signal
        # We must match GPT-2's embedding dimension
        self.countdown_emb = CountdownEmbedding(768, max_len=config.max_len + 5)

        # 4. Adapter / Projection
        # We need to inform GPT-2 about the image.
        # Simple strategy: Prefix Tuning (concat image tokens) or Cross Attention?
        # GPT-2 is Decoder-only, standard Cross-Attention isn't default.
        # We will use "Visual Prefix": Prepend visual tokens to the sequence.

    def forward(self, images, input_ids, target_lengths):
        B, L = input_ids.shape
        device = images.device

        # --- 1. Get Visual Features ---
        # Shape: [B, 49, 768]
        visual_embeds = self.vision_encoder(images)

        # --- 2. Prepare Text Embeddings ---
        # We leverage GPT-2's internal embeddings (wte) and positional (wpe)
        # inputs_embeds = Word + Pos + Countdown

        # Word Embeddings from GPT-2
        word_embeds = self.gpt2.transformer.wte(input_ids)

        # Positional Embeddings from GPT-2
        # GPT-2 uses absolute positions 0..L
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.gpt2.transformer.wpe(positions)

        # Countdown Embeddings (Our Custom Logic)
        count_embeds = self.countdown_emb(positions, target_lengths)

        # Combine Text Signal
        text_inputs_embeds = word_embeds + pos_embeds + count_embeds

        # --- 3. Fuse Vision + Text ---
        # We prepend the visual features to the text embeddings.
        # The model sees: [Image_Patch_1, ..., Image_Patch_49, Word_1, ..., Word_L]
        # Note: We must ensure we don't calculate loss on image tokens.

        combined_embeds = torch.cat([visual_embeds, text_inputs_embeds], dim=1)

        # --- 4. Forward Pass through GPT-2 ---
        # We generate output. GPT-2 handles the rest.
        output = self.gpt2(inputs_embeds=combined_embeds)
        logits = output.logits

        # The logits include predictions for the Image tokens too (at the start).
        # We only care about the predictions for the text part.
        # Output shape: [B, 49 + L, Vocab]
        # We want the last L tokens.
        text_logits = logits[:, 49:, :]

        return text_logits
