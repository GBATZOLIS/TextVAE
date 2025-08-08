# models/vision_encoder.py
import torch
import torch.nn as nn
from transformers import SwinModel, GPT2LMHeadModel, GPT2Config
from torch.utils.checkpoint import checkpoint


class VisionEncoder(nn.Module):
    """
    Encodes an image into a continuous, differentiable text representation.
    This consists of a vision backbone (Swin) and a text generation head (GPT-2 style).
    """

    def __init__(self, swin_model_name: str, max_text_length: int = 256):
        super().__init__()
        self.max_text_length = max_text_length

        # 1. Vision Backbone
        self.swin = SwinModel.from_pretrained(swin_model_name)

        # 2. Text Generation Head (custom config for cross-attention)
        text_config = GPT2Config.from_pretrained("gpt2")
        text_config.add_cross_attention = True
        text_config.use_cache = False
        self.text_head = GPT2LMHeadModel(config=text_config)

        # 3. Projection layer to match Swin output to GPT-2 input dimensions
        self.projection = nn.Linear(self.swin.config.hidden_size, text_config.n_embd)

        # 4. A learned "start-of-sequence" embedding
        self.start_token_emb = nn.Parameter(torch.randn(1, 1, text_config.n_embd))

    def _generation_step(self, current_embeddings, encoder_hidden_states):
        """A helper function for the generation step to use with checkpointing."""
        outputs = self.text_head(
            inputs_embeds=current_embeddings,
            encoder_hidden_states=encoder_hidden_states,
        )
        return outputs.logits[:, -1, :]

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that generates soft text representations.

        Args:
            image (torch.Tensor): Input image batch (B, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits (B, seq_len, vocab_size): Differentiable logits for the plausibility loss.
                - soft_embeddings (B, seq_len, embed_dim): Differentiable embeddings for the diffusion decoder.
        """
        batch_size = image.shape[0]

        # Get visual features from Swin Transformer
        visual_features = self.swin(image).last_hidden_state
        encoder_hidden_states = self.projection(
            visual_features
        )  # (B, num_patches, embed_dim)

        # Prepare for autoregressive generation
        # We start with a learnable start token
        current_embeddings = self.start_token_emb.expand(batch_size, -1, -1)

        all_logits = []

        # Autoregressively generate the sequence of logits
        for _ in range(self.max_text_length):
            # Pass visual features as cross-attention context
            next_token_logits = checkpoint(
                self._generation_step,
                current_embeddings,
                encoder_hidden_states,
                use_reentrant=False,
            )

            all_logits.append(next_token_logits)

            # For the next step, use the embedding of the predicted token.
            # To keep it differentiable, we use the "soft" embedding by taking a weighted
            # average of the embedding matrix, weighted by the softmax of the logits.
            word_embedding_matrix = (
                self.text_head.get_input_embeddings().weight.t()
            )  # (embed_dim, vocab_size)
            soft_next_embeddings = torch.matmul(
                torch.softmax(next_token_logits, dim=-1), word_embedding_matrix.t()
            ).unsqueeze(
                1
            )  # (B, 1, embed_dim)

            # Append the new soft embedding for the next iteration
            current_embeddings = torch.cat(
                [current_embeddings, soft_next_embeddings], dim=1
            )

        # Combine all logits into a single tensor
        logits = torch.stack(all_logits, dim=1)  # (B, seq_len, vocab_size)

        # The final output embeddings are the ones we generated (excluding start token)
        soft_embeddings = current_embeddings[:, 1:, :]  # (B, seq_len, embed_dim)

        return logits, soft_embeddings
