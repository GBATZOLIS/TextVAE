import torch
import torch.nn as nn
from diffusers import (
    PixArtAlphaPipeline,
    PixArtTransformer2DModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import T5EncoderModel, T5Tokenizer
from peft import get_peft_model, LoraConfig


class SemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        print("Loading T5-XXL Tokenizer and Encoder...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.model_id, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            config.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.text_encoder.requires_grad_(False)

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            config.model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.vae.requires_grad_(False)

        print("Loading Noise Scheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.model_id, subfolder="scheduler"
        )

        print("Loading PixArt DiT and applying LoRA...")
        base_transformer = PixArtTransformer2DModel.from_pretrained(
            config.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # FIX 1: Lower lora_alpha to 64.
        # A 1.0 scaling ratio (64/64) prevents activation explosion on all-linear targets.
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            lora_dropout=0.05,  # <-- ADDED: Crucial regularization for "all-linear" capacity
            bias="none",
        )
        self.transformer = get_peft_model(base_transformer, lora_config)

    def forward(self, images, input_ids, attention_mask):
        bsz = images.shape[0]
        self.text_encoder.eval()
        self.vae.eval()

        # --- TRUE CONDITIONING DROPOUT FOR CFG ---
        if self.training:
            input_ids = input_ids.clone()
            attention_mask = attention_mask.clone()
            drop_mask = torch.rand(bsz, device=self.device) < 0.1

            if drop_mask.any():
                # FIX 1: Provide the exact empty string encoding instead of zeroing the mask!
                # This ensures the model learns the exact same "unconditional" vector the pipeline uses at inference.
                empty_encoded = self.tokenizer(
                    "",
                    padding="max_length",
                    # max_length=input_ids.shape[1], # <-- Match current batch length
                    max_length=self.config.max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids[drop_mask] = empty_encoded.input_ids.to(self.device)
                attention_mask[drop_mask] = empty_encoded.attention_mask.to(self.device)

        with torch.no_grad():
            latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            prompt_embeds = self.text_encoder(
                input_ids.to(self.device), attention_mask=attention_mask.to(self.device)
            )[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        added_cond_kwargs = {
            "resolution": torch.full((bsz,), self.config.img_size, device=self.device),
            "aspect_ratio": torch.full((bsz,), 1.0, device=self.device),
        }

        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds.to(noisy_latents.dtype),
            encoder_attention_mask=attention_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        # --- MIN-SNR-GAMMA WEIGHTING ---
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5

        snr = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2
        min_snr_gamma = 5.0

        # <-- FIXED: This line was missing, breaking the loss multiplier!
        snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr

        # 7. Compute Loss
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), noise.float(), reduction="none"
        )
        loss = loss.mean(dim=[1, 2, 3])
        loss = (loss * snr_weight).mean()

        return loss

    @torch.no_grad()
    def generate(self, texts, num_inference_steps=20, guidance_scale=4.5):

        # FIX 3: DDPMScheduler is mathematically incapable of 20-step generation.
        # We MUST swap it for DPMSolver at inference to prevent "painted" discretization artifacts!
        eval_scheduler = DPMSolverMultistepScheduler.from_config(
            self.noise_scheduler.config
        )

        pipeline = PixArtAlphaPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=eval_scheduler,
        ).to(self.device)

        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            prompt=texts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,  # Keep your scale high for accuracy
            guidance_rescale=0.7,  # THE FIX: Add this! Usually 0.7 works magic.
            height=self.config.img_size,
            width=self.config.img_size,
            max_sequence_length=self.config.max_len,
        ).images

        return images
