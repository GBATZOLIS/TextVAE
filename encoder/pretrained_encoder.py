import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, Dinov2Model
from peft import get_peft_model, LoraConfig, TaskType


class VisualMapper(nn.Module):
    def __init__(self, visual_dim=1024, gpt_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, gpt_dim),
            nn.LayerNorm(gpt_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gpt_dim, gpt_dim),
        )

    def forward(self, x):
        return self.proj(x)


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


class PlanningGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Vision: DINOv2 ViT-Large
        self.vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-large")
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 2. Visual Mapper
        self.visual_mapper = VisualMapper(visual_dim=1024, gpt_dim=768)

        # 3. Language: GPT-2 with Cross-Attention Enabled
        gpt2_config = GPT2Config.from_pretrained("gpt2")
        gpt2_config.add_cross_attention = (
            True  # THE FIX: Injects cross-attention into every block
        )

        # Load base model (ignore_mismatched_sizes is needed because the new cross-attention weights are randomly initialized)
        base_gpt2 = GPT2LMHeadModel.from_pretrained(
            "gpt2", config=gpt2_config, ignore_mismatched_sizes=True
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"],
            # THE FIX: Tell PEFT to keep the newly initialized cross-attention layers fully trainable
            modules_to_save=["crossattention"],
            fan_in_fan_out=True,
        )
        self.gpt2 = get_peft_model(base_gpt2, peft_config)

        # 4. Countdown Mechanism
        self.countdown_emb = CountdownEmbedding(768, max_len=config.max_len + 5)

    def forward(self, images, input_ids, target_lengths, attention_mask=None):
        B, L = input_ids.shape
        device = images.device

        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=images)
            raw_visual = outputs.last_hidden_state[:, 1:, :]

        visual_embeds = self.visual_mapper(raw_visual)

        word_embeds = self.gpt2.base_model.model.transformer.wte(input_ids)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        count_embeds = self.countdown_emb(positions, target_lengths)

        # NO MORE CONCATENATION. Text is its own pure sequence.
        text_inputs_embeds = word_embeds + count_embeds

        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.long, device=device)

        # THE FIX: Pass visual_embeds as encoder_hidden_states.
        # GPT-2 will automatically route this to the cross-attention layers.
        outputs = self.gpt2(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_embeds,
            # encoder_attention_mask defaults to all 1s (which is correct for DINO patches)
        )

        # NO MORE SLICING ([:, 256:, :]). The sequence length is exactly the text length.
        return outputs.logits
