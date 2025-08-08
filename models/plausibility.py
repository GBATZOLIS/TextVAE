# models/plausibility.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class PlausibilityModule(nn.Module):
    """
    A non-trainable module that uses a frozen LLM to assess the plausibility
    of generated text representations (logits).
    """

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Ensure it's in eval mode

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, generated_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates a plausibility loss using KL-Divergence. This forces the
        distribution of the generated logits to match the distribution
        predicted by the powerful frozen LLM.

        Args:
            generated_logits (torch.Tensor): The logit tensor from the VisionEncoder.

        Returns:
            torch.Tensor: A scalar KL-divergence loss.
        """
        with torch.no_grad():
            # To get target logits from the LLM, we need to feed it the "hard" tokens
            # that would have been generated. We do this without gradients.
            generated_ids = torch.argmax(generated_logits, dim=-1)

            # The LLM predicts the next token's logits for each position in the sequence.
            target_outputs = self.model(input_ids=generated_ids)
            target_logits = target_outputs.logits.detach()  # (B, seq_len, vocab_size)

        # Calculate KL Divergence loss.
        # We want the distribution of our model's logits to be close to the
        # distribution of the frozen LLM's logits.

        # Use log_softmax for numerical stability, as required by kl_div.
        log_probs_generated = F.log_softmax(generated_logits, dim=-1)
        probs_target = F.softmax(target_logits, dim=-1)

        # The KL divergence loss. `reduction='batchmean'` averages the loss over the batch.
        kl_loss = F.kl_div(
            log_probs_generated, probs_target, reduction="batchmean", log_target=False
        )

        return kl_loss
