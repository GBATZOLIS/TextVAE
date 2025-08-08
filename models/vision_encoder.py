# models/vision_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import cast, List


class VLMEncoder(nn.Module):
    """
    Encodes an image using a PEFT-adapted VLM for training and generates text for inference.
    """

    def __init__(self, cfg, device: torch.device):
        super().__init__()
        self.device = device
        self.cfg = cfg

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.processor = LlavaNextProcessor.from_pretrained(cfg.models.vlm_id)
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.models.vlm_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # --- PROMPT ENGINEERING: RESTORE THIS SECTION ---
        # This line was missing. It defines the prompt template for the VLM.
        self.prompt_text = "[INST] <image>\nDescribe this image in detail. [/INST]"
        # -------------------------------------------------

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **For Training:** Performs a forward pass to get soft embeddings and logits.
        """
        inputs = self.processor(
            text=self.prompt_text, images=pixel_values, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values.to(torch.float16),
            image_sizes=inputs.image_sizes,
            return_dict=True,
        )

        logits = outputs.logits

        # --- CRITICAL FIX HERE ---
        # Cast logits to float32 before the sensitive softmax operation to prevent
        # numerical overflow (inf/NaN) in fp16.
        gumbel_output = F.gumbel_softmax(
            logits.to(torch.float32),
            tau=self.cfg.training.gumbel_temperature,
            hard=False,
            dim=-1,
        )
        # -------------------------

        language_model_embeddings = self.model.get_input_embeddings().weight
        soft_embeddings = torch.einsum(
            "bse,ed->bsd", gumbel_output, language_model_embeddings
        )

        return soft_embeddings, logits

    @torch.no_grad()
    def generate_text(self, pixel_values: torch.Tensor) -> list[str]:
        self.eval()
        # Corrected typo here from "tentsors" to "tensors"
        inputs = self.processor(
            text=self.prompt_text, images=pixel_values, return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values.to(torch.float16),
            image_sizes=inputs.image_sizes,
            max_new_tokens=self.cfg.training.max_new_tokens,
        )

        return cast(
            List[str],
            self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            ),
        )
