import torch
from controlnet_aux import CannyDetector
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image
import requests
import argparse
import os
import logging
from typing import cast

# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- 2. Enhanced Model Configuration ---
MODELS_CONFIG = {
    "vlm": {
        "llava-next": {
            "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
            "prompt": """[INST] <image>
Provide an exhaustive, encyclopedic description of the provided image, suitable for a forensic visual analysis. Break down the output into the following sections using markdown headings:

## Primary Subject Analysis
- **Identification:** What is the primary subject?
- **Appearance & Attire:** Describe its physical characteristics, clothing, texture, and condition in minute detail.
- **Pose & Expression:** Detail its posture, gaze, facial expression, and implied emotion or action.

## Environmental Context
- **Foreground, Midground, Background:** Detail every element in each spatial plane.
- **Setting & Location:** Identify the probable location (e.g., urban street, forest, studio).
- **Props & Objects:** List and describe every secondary object, its state, and its relation to the main subject.

## Photographic & Artistic Properties
- **Medium & Style:** (e.g., digital photograph, oil painting, 3D render; photorealistic, impressionistic, surreal).
- **Composition:** Analyze the framing (e.g., rule of thirds, centered, dutch angle), shot type (e.g., close-up, wide shot), and perspective.
- **Lighting:** Describe the quality, color, and direction of the main light source(s) and ambient light. Note the properties of shadows (hard/soft, long/short).
- **Color Palette:** Describe the dominant, secondary, and accent colors. Mention the overall color harmony and mood (e.g., monochromatic, analogous, complementary; vibrant, muted, somber).
[/INST]""",
        }
    },
    "t2i": {"sdxl": {"model_id": "stabilityai/stable-diffusion-xl-base-1.0"}},
}

# --- 3. Self-Contained Functions ---


@torch.no_grad()
def compress_image_to_text(
    image: Image.Image, vlm_config: dict, checkpoint_path: str | None
) -> str:
    """Loads VLM, generates a single detailed description, and unloads VLM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"--- Starting Compression Phase on {device} ---")

    logging.info(f"Loading VLM: {vlm_config['model_id']}")
    processor = LlavaNextProcessor.from_pretrained(vlm_config["model_id"])
    model = LlavaNextForConditionalGeneration.from_pretrained(
        vlm_config["model_id"], torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)

    if checkpoint_path:
        logging.info(f"Loading fine-tuned VLM weights from: {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device).get(
                "vlm_encoder_state_dict", {}
            )
        )
    model.eval()

    prompt_template = vlm_config["prompt"]
    inputs = processor(text=prompt_template, images=image, return_tensors="pt").to(
        device
    )
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    full_description = (
        processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        .split("[/INST]")[-1]
        .strip()
    )

    logging.info("Unloading VLM from GPU memory...")
    del model
    del processor
    torch.cuda.empty_cache()

    return cast(str, full_description)


@torch.no_grad()
def decompress_text_to_image(
    prompt: str, original_image: Image.Image, t2i_config: dict, steps: int
) -> Image.Image:
    """Loads T2I models and generates an image from the provided prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"--- Starting Decompression Phase on {device} ---")

    logging.info(f"Loading T2I Pipeline: {t2i_config['model_id']}")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    ).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        t2i_config["model_id"], controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    canny_detector = CannyDetector()

    canny_image = canny_detector(original_image, low_threshold=100, high_threshold=200)
    reconstructed_image = pipe(
        prompt=prompt,
        negative_prompt="low quality, worst quality, deformed, blurry",
        image=canny_image,
        num_inference_steps=steps,
        guidance_scale=7.0,
    ).images[0]

    logging.info("Unloading T2I pipeline from GPU memory...")
    del pipe
    del controlnet
    torch.cuda.empty_cache()

    return reconstructed_image


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for the Semantic Image Codec."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the fine-tuned VLM Encoder checkpoint."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path or URL to the input image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the output.",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of diffusion inference steps."
    )
    args = parser.parse_args()

    # setup_logging()

    try:
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if args.image.startswith("http"):
            input_image = Image.open(requests.get(args.image, stream=True).raw).convert(
                "RGB"
            )
        else:
            input_image = Image.open(os.path.expanduser(args.image)).convert("RGB")

        # 1. Compress image to a single, detailed text description
        full_description = compress_image_to_text(
            image=input_image,
            vlm_config=MODELS_CONFIG["vlm"]["llava-next"],
            checkpoint_path=args.checkpoint,
        )

        logging.info(f"\n--- Full Generated Description ---\n{full_description}")

        with open(
            os.path.join(output_dir, "full_description.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(full_description)

        # 2. Decompress using the full description as the prompt
        reconstructed_image = decompress_text_to_image(
            prompt=full_description,  # <-- Use the full description here
            original_image=input_image,
            t2i_config=MODELS_CONFIG["t2i"]["sdxl"],
            steps=args.steps,
        )

        logging.info(f"Saving final images to {output_dir}...")
        input_image.resize(reconstructed_image.size).save(
            os.path.join(output_dir, "original.png")
        )
        reconstructed_image.save(os.path.join(output_dir, "reconstructed.png"))

        logging.info("Inference complete.")

    except Exception as e:
        logging.error(
            f"An unrecoverable error occurred during the pipeline: {e}", exc_info=True
        )
