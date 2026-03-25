import torch
import torch.nn as nn
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig


class FluxSemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        model_id = "black-forest-labs/FLUX.1-schnell"  # Or -dev if you want the 50-step version

        print("Loading Tokenizers & Encoders (CLIP + T5)...")
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = T5Tokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        print("Loading FLUX VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.vae.requires_grad_(False)

        print("Loading FLUX Scheduler (Flow Matching)...")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        print("Loading FLUX Transformer and applying LoRA...")
        base_transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # Enable Gradient Checkpointing to survive the 12B parameter footprint!
        base_transformer.enable_gradient_checkpointing()

        # FLUX uses a completely different attention architecture (MMDiT).
        # We target the specific linear layers used in its joint attention blocks.
        lora_config = LoraConfig(
            r=16,  # Lower rank because the model is so massive
            lora_alpha=16,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
        )
        self.transformer = get_peft_model(base_transformer, lora_config)

    def encode_prompt(self, captions):
        # FLUX requires both CLIP (for global style) and T5 (for dense details)
        with torch.no_grad():
            # 1. CLIP Encoding (Pooled)
            text_inputs_1 = self.tokenizer_1(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            pooled_prompt_embeds = self.text_encoder(
                text_inputs_1.input_ids, output_hidden_states=False
            ).pooler_output

            # 2. T5 Encoding (Sequence)
            text_inputs_2 = self.tokenizer_2(
                captions,
                padding="max_length",
                max_length=self.config.max_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_embeds = self.text_encoder_2(text_inputs_2.input_ids)[0]

        return prompt_embeds, pooled_prompt_embeds

    def prepare_flux_latents(self, images):
        # VAE Encoding
        with torch.no_grad():
            latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
            latents = (
                latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

        bsz, channels, height, width = latents.shape

        # FLUX requires "packing" the 2D latents into a 1D sequence for its transformer
        latents = latents.view(bsz, channels, -1).transpose(1, 2)  # [B, H*W, C]

        # FLUX also requires explicit positional ID tensors for the image patches
        img_ids = torch.zeros((height, width, 3), device=self.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width)[None, :]
        img_ids = (
            img_ids.view(-1, 3)
            .unsqueeze(0)
            .repeat(bsz, 1, 1)
            .to(self.device, dtype=latents.dtype)
        )

        return latents, img_ids

    def forward(self, images, captions):
        # NOTE: Your dataloader now passes raw string `captions` instead of pre-tokenized ids,
        # so we can tokenize for both CLIP and T5 inside the model.
        bsz = images.shape[0]

        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()

        # 1. Get Latents and Image IDs
        latents, img_ids = self.prepare_flux_latents(images)

        # 2. Get Dual Text Embeddings
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(captions)

        # FLUX also needs Text IDs for positional routing
        txt_ids = torch.zeros(
            (bsz, prompt_embeds.shape[1], 3), device=self.device, dtype=latents.dtype
        )

        # 3. RECTIFIED FLOW MATH (No more DDPM!)
        noise = torch.randn_like(latents)

        # Sample random timesteps between 0 and 1 (Flow Matching uses 0->1 rather than 1000->0)
        u = torch.rand((bsz,), device=self.device).to(latents.dtype)

        # The equation for Flow Matching blending: x_t = (1 - t) * image + t * noise
        u_expanded = u.unsqueeze(1).unsqueeze(2)
        noisy_latents = (1.0 - u_expanded) * latents + u_expanded * noise

        # The target in Flow Matching is the vector pointing from the image to the noise
        target = noise - latents

        # 4. Forward Pass
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=u,
            guidance=torch.full(
                (bsz,), 3.5, device=self.device, dtype=latents.dtype
            ),  # FLUX uses embedded guidance
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

        # 5. MSE Loss against the Vector Field
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )
        return loss

    @torch.no_grad()
    def generate(self, texts, num_inference_steps=4):
        # If using FLUX.1-dev, use ~25 steps. For FLUX.1-schnell, 4 steps is SOTA!

        pipeline = FluxPipeline(
            tokenizer=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            prompt=texts,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # FLUX-Schnell does not use CFG! Perfect prompt adherence by default.
            height=self.config.img_size,
            width=self.config.img_size,
            max_sequence_length=self.config.max_len,
        ).images

        return images
