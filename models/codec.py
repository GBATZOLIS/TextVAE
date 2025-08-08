# models/codec.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from typing import cast
from peft import get_peft_model, LoraConfig
from PIL import Image, ImageDraw

from config import AppConfig


class Codec(nn.Module):
    def __init__(self, cfg: AppConfig, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- 1. VLM ENCODER (for training AND inference captioning) ---
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.vlm_processor = LlavaNextProcessor.from_pretrained(
            cfg.models.training_vlm_id
        )
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.models.training_vlm_id,
            quantization_config=quant_config,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.vlm_model = get_peft_model(vlm_model, lora_config)
        self.vlm_model.gradient_checkpointing_enable()
        self.vlm_prompt = "[INST] <image>\nDescribe this image in detail. [/INST]"

        # --- 2. LAYOUT GENERATOR (LLM) - LOADED ON CPU TO SAVE VRAM ---
        self.layout_tokenizer = AutoTokenizer.from_pretrained(
            cfg.models.text_segmenter_id
        )
        self.layout_model = AutoModelForCausalLM.from_pretrained(
            cfg.models.text_segmenter_id, torch_dtype=torch.float16, device_map="cpu"
        )

        # --- 3. DECODER COMPONENTS ---
        self.vae = AutoencoderKL.from_pretrained(
            cfg.models.t2i_id, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.models.t2i_id, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(
            cfg.models.t2i_id, subfolder="scheduler"
        )
        self.controlnet = ControlNetModel.from_pretrained(
            cfg.models.controlnet_id, torch_dtype=torch.float16
        ).to(device)

        self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.models.t2i_id,
            vae=self.vae,
            unet=self.unet,
            scheduler=self.scheduler,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        # --- 4. TRAINABLE PROJECTION LAYER ---
        self.projection = nn.Linear(
            self.vlm_model.config.text_config.hidden_size,
            self.unet.config.cross_attention_dim,
            dtype=torch.float16,
        ).to(device)

        # --- 5. FREEZE UNUSED MODELS ---
        self.layout_model.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict:
        inputs = self.vlm_processor(
            text=self.vlm_prompt, images=images, return_tensors="pt"
        ).to(self.vlm_model.device)
        outputs = self.vlm_model(**inputs, output_hidden_states=True)
        vlm_embeds = outputs.hidden_states[-1]
        projected_embeds = self.projection(vlm_embeds)
        with torch.no_grad():
            latents = (
                self.vae.encode(
                    images.to(self.device, dtype=torch.float16)
                ).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        predicted_noise = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=projected_embeds
        ).sample
        loss = F.mse_loss(predicted_noise, noise)
        return {"loss": loss}

    @torch.no_grad()
    def encode(self, image_pil: Image.Image) -> str:
        """Encodes an image into text using the trained VLM."""
        inputs = self.vlm_processor(
            text=self.vlm_prompt, images=image_pil, return_tensors="pt"
        ).to(self.vlm_model.device)
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=256)
        return cast(
            str,
            self.vlm_processor.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True
            )[0].strip(),
        )

    @torch.no_grad()
    def _get_layout_image(self, description: str) -> Image.Image:
        """Uses an LLM (temporarily moved to GPU) to generate a layout image."""
        prompt = f'Parse the following description and return a JSON list of objects with normalized bounding boxes (e.g., [x1, y1, x2, y2]). Description: "{description}"\nJSON:'

        # --- CRITICAL MEMORY MANAGEMENT ---
        # 1. Move the large UNet to CPU to make space
        self.unet.to("cpu")
        torch.cuda.empty_cache()

        # 2. Move layout model to GPU for inference
        self.layout_model.to(self.device)

        inputs = self.layout_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.layout_model.generate(**inputs, max_new_tokens=256)
        response_text = self.layout_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        # 3. Move layout model back to CPU to free VRAM
        self.layout_model.to("cpu")
        torch.cuda.empty_cache()

        # 4. Move UNet back to GPU for the diffusion step
        self.unet.to(self.device)
        # ------------------------------------

        layout_image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(layout_image)
        try:
            json_str = response_text[
                response_text.find("[") : response_text.rfind("]") + 1
            ]
            objects = json.loads(json_str)
            for obj in objects:
                box = obj.get("box", [])
                if len(box) == 4:
                    x1, y1, x2, y2 = [coord * 512 for coord in box]
                    draw.rectangle([x1, y1, x2, y2], outline="black", width=3)
        except Exception as e:
            print(f"Warning: Could not parse layout. Using default. Error: {e}")
            draw.rectangle([50, 50, 462, 462], outline="black", width=3)

        return layout_image

    @torch.no_grad()
    def _get_conditioning_embeds(self, text: str) -> torch.Tensor:
        """Uses the TRAINED VLM to get conditioning embeddings from text."""
        # Use a dummy black image as the VLM requires an image input
        dummy_image = Image.new(
            "RGB",
            (self.cfg.data.image_resolution, self.cfg.data.image_resolution),
            "black",
        )
        full_prompt = f"[INST] <image>\n{text} [/INST]"
        inputs = self.vlm_processor(
            text=full_prompt, images=dummy_image, return_tensors="pt"
        ).to(self.vlm_model.device)
        outputs = self.vlm_model(**inputs, output_hidden_states=True)
        vlm_embeds = outputs.hidden_states[-1]
        return self.projection(vlm_embeds)

    @torch.no_grad()
    def decode_long_context(self, text_prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt_embeds = self._get_conditioning_embeds(text_prompt)
        # Use the standard SD pipe logic, which is part of controlnet_pipe but without an image
        return self.controlnet_pipe(
            prompt_embeds=prompt_embeds, generator=generator
        ).images[0]

    @torch.no_grad()
    def decode_controlnet(self, text_prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt_embeds = self._get_conditioning_embeds(text_prompt)
        layout_image = self._get_layout_image(text_prompt)
        control_image = self.canny_detector(
            layout_image, low_threshold=100, high_threshold=200
        )
        return self.controlnet_pipe(
            prompt_embeds=prompt_embeds, image=control_image, generator=generator
        ).images[0]

    @torch.no_grad()
    def decode_hybrid(self, text_prompt: str, seed: int) -> Image.Image:
        return self.decode_controlnet(text_prompt, seed)
