import torch
import torch.nn as nn
from diffusers import (
    PixArtAlphaPipeline,
    PixArtTransformer2DModel,
    AutoencoderKL,
    DDPMScheduler,
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
        # Freeze T5
        self.text_encoder = T5EncoderModel.from_pretrained(
            config.model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        self.text_encoder.requires_grad_(False)

        print("Loading VAE...")
        # Freeze VAE (Diffusers expects [-1, 1] normalized inputs)
        self.vae = AutoencoderKL.from_pretrained(
            config.model_id, subfolder="vae", torch_dtype=torch.float16
        )
        self.vae.requires_grad_(False)

        print("Loading Noise Scheduler...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.model_id, subfolder="scheduler"
        )

        print("Loading PixArt DiT and applying LoRA...")
        base_transformer = PixArtTransformer2DModel.from_pretrained(
            config.model_id, subfolder="transformer"
        )

        # Apply LoRA to the attention layers of the DiT
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        self.transformer = get_peft_model(base_transformer, lora_config)

    def forward(self, images, input_ids, attention_mask):
        """
        Training pass: Encodes image to latents, adds noise, encodes text, and predicts noise.
        """
        bsz = images.shape[0]

        with torch.no_grad():
            # 1. Encode Images to Latents
            # Move images to VAE's dtype (fp16) to save memory
            latents = self.vae.encode(images.to(self.vae.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # 2. Encode Text using T5
            prompt_embeds = self.text_encoder(
                input_ids.to(self.device), attention_mask=attention_mask.to(self.device)
            )[0]

        # 3. Sample Noise & Timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        # 4. Add Noise (Forward Diffusion)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 5. PixArt specific condition kwargs (Resolution & Aspect Ratio)
        added_cond_kwargs = {
            "resolution": torch.full((bsz,), self.config.img_size, device=self.device),
            "aspect_ratio": torch.full((bsz,), 1.0, device=self.device),
        }

        # 6. Predict Noise
        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=prompt_embeds.to(noisy_latents.dtype),
            encoder_attention_mask=attention_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # 7. Compute Loss
        loss = nn.functional.mse_loss(
            model_pred.float(), noise.float(), reduction="mean"
        )
        return loss

    @torch.no_grad()
    def generate(self, texts, num_inference_steps=20, guidance_scale=4.5):
        """
        Inference pass: Dynamically builds the pipeline and generates images from text.
        """
        # We construct the pipeline on the fly using our LoRA-adapted transformer
        pipeline = PixArtAlphaPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        # Free up memory before generation
        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            prompt=texts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.config.img_size,
            width=self.config.img_size,
        ).images

        return images
