# models/vision_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class VLMEncoder(nn.Module):
    """
    Encodes an image into a sequence of differentiable "soft" embeddings
    using a full Vision-Language Model (VLM).
    """

    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        self.device = device

        # --- Model Loading ---
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.device)

        # --- Prompt Engineering ---
        self.prompt_text = (
            "[INST] <image>\nA highly detailed, descriptive caption: [/INST]"
        )
        self.input_ids = self.processor(
            text=self.prompt_text, return_tensors="pt"
        ).input_ids.to(self.device)
        self.prompt_len = self.input_ids.shape[1]

    def forward(
        self, pixel_values: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass of the encoder."""
        # --- Logit Generation ---
        batch_input_ids = self.input_ids.repeat(pixel_values.shape[0], 1)
        # The VLM's generate function is complex; for training, we use a direct forward pass
        outputs = self.model(
            input_ids=batch_input_ids, pixel_values=pixel_values, return_dict=True
        )
        # We only care about the logits for the *generated* part of the sequence.
        logits = outputs.logits[:, self.prompt_len :, :]

        # --- Differentiable Sampling ---
        # Use PyTorch's built-in Gumbel-Softmax for stable, differentiable sampling.
        gumbel_output = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)

        # --- Create Soft Embeddings ---
        # Project the Gumbel output back into the embedding space.
        language_model_embeddings = self.model.get_input_embeddings().weight
        soft_embeddings = torch.einsum(
            "bse,ed->bsd", gumbel_output, language_model_embeddings
        )

        return soft_embeddings, logits
