import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, Dinov2Model
from peft import get_peft_model, LoraConfig, TaskType


class PerceiverResampler(nn.Module):
    """
    Compresses 257 raw DINOv2 patches (including CLS) into 32 dense
    semantic tokens using Cross-Attention.
    """

    def __init__(self, visual_dim=1024, gpt_dim=768, num_queries=32, depth=3, heads=12):
        super().__init__()
        self.num_queries = num_queries

        # Truncated Normal initialization prevents gradient saturation
        # and massive outliers in early training steps.
        self.latents = nn.Parameter(torch.empty(1, num_queries, gpt_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.proj_in = nn.Linear(visual_dim, gpt_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=gpt_dim, num_heads=heads, batch_first=True
                        ),
                        nn.LayerNorm(gpt_dim),
                        nn.MultiheadAttention(
                            embed_dim=gpt_dim, num_heads=heads, batch_first=True
                        ),
                        nn.LayerNorm(gpt_dim),
                        nn.Sequential(
                            nn.Linear(gpt_dim, gpt_dim * 4),
                            nn.GELU(),
                            nn.Linear(gpt_dim * 4, gpt_dim),
                        ),
                        nn.LayerNorm(gpt_dim),
                    ]
                )
            )

    def forward(self, x):
        # x is DINOv2 features: [B, 257, 1024]
        x = self.proj_in(x)
        b = x.shape[0]

        latents = self.latents.repeat(b, 1, 1)

        for cross_attn, norm1, self_attn, norm2, ff, norm3 in self.layers:
            attn_out, _ = cross_attn(query=latents, key=x, value=x)
            latents = norm1(latents + attn_out)

            attn_out, _ = self_attn(query=latents, key=latents, value=latents)
            latents = norm2(latents + attn_out)

            latents = norm3(latents + ff(latents))

        return latents  # [B, 32, 768]


class CountdownEmbedding(nn.Module):
    """
    Discrete Absolute Time Embedding: Embeds the exact number of tokens
    remaining. This preserves local grammatical structure beautifully.
    """

    def __init__(self, dim=768, max_len=512):
        super().__init__()
        # +1 to handle hitting absolute 0
        self.embedding = nn.Embedding(max_len + 1, dim)
        self.max_len = max_len

    def forward(self, current_positions, target_lengths):
        if target_lengths.dim() == 1:
            target_lengths = target_lengths.unsqueeze(1)

        # Exact integer distance to the target length
        distance = target_lengths - current_positions

        # Clamp between 0 and max_len to prevent out-of-bounds crashes
        # if the model hallucinates past the target length during generation.
        distance = torch.clamp(distance, min=0, max=self.max_len)
        return self.embedding(distance)


class PlanningGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Vision: DINOv2 ViT-Large
        self.vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-large")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 2. Visual Mapper: Perceiver Resampler
        self.num_visual_queries = 32
        self.visual_mapper = PerceiverResampler(
            visual_dim=1024, gpt_dim=768, num_queries=self.num_visual_queries
        )

        # 3. Language: GPT-2 via LoRA
        base_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
            fan_in_fan_out=True,
        )
        self.gpt2 = get_peft_model(base_gpt2, peft_config)

        # 4. Countdown Mechanism: Restored to Discrete Integer Embedding
        # Give it a small buffer above config.max_len just in case
        self.countdown_emb = CountdownEmbedding(dim=768, max_len=config.max_len + 5)

    def forward(self, images, input_ids, target_lengths, attention_mask=None):
        B, L = input_ids.shape
        device = images.device

        # Extract Visuals
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=images)
            # Retain the CLS token for global context!
            raw_visual = outputs.last_hidden_state

        # Compress to 32 queries
        visual_embeds = self.visual_mapper(raw_visual)

        # Process Text
        word_embeds = self.gpt2.base_model.model.transformer.wte(input_ids)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Restore standard GPT-2 positional embeddings
        wpe = self.gpt2.base_model.model.transformer.wpe(positions)

        # Add the discrete countdown
        count_embeds = self.countdown_emb(positions, target_lengths)

        # Tri-Embedding Setup
        text_inputs_embeds = word_embeds + wpe + count_embeds

        # Concat [B, 32, 768] with [B, L, 768]
        combined_embeds = torch.cat([visual_embeds, text_inputs_embeds], dim=1)

        # Build the proper joint attention mask to prevent padding poisoning
        if attention_mask is None:
            text_mask = torch.ones((B, L), dtype=torch.long, device=device)
        else:
            text_mask = attention_mask

        visual_mask = torch.ones(
            (B, self.num_visual_queries), dtype=torch.long, device=device
        )
        full_attention_mask = torch.cat([visual_mask, text_mask], dim=1)

        # Forward pass through GPT-2
        outputs = self.gpt2(
            inputs_embeds=combined_embeds, attention_mask=full_attention_mask
        )

        # Return only the text logits (slice off the 32 visual query predictions)
        return outputs.logits[:, self.num_visual_queries :, :]
