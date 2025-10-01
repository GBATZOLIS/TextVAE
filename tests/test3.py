# generate_long_description.py

"""
Generates a long-form, 500-word description of an image by chaining a
vision model with a powerful language model.

1.  VISION-TO-TEXT: A captioning model extracts a factual, one-sentence
                       summary from the image.
2.  TEXT-TO-TEXT:   A Large Language Model (LLM) receives the simple caption
                       and a detailed prompt, instructing it to elaborate into
                       a long, descriptive narrative.
3.  TEXT-TO-IMAGE:  The resulting long description is fed to a diffusion model
                       to generate the reconstructed image.
"""
import torch
import requests
from PIL import Image
import logging

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model IDs ---
# 1. Vision Model for initial caption
IMG2TEXT_MODEL = "Salesforce/blip-image-captioning-large"

# 2. Powerful LLM for creative elaboration.
#    Using a smaller, instruction-tuned model is recommended.
#    Mistral-7B is a great choice.
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# 3. Diffusion model for final image reconstruction
TEXT2IMG_MODEL = "runwayml/stable-diffusion-v1-5"


def main():
    # --- Load a sample image ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        logger.info(f"Downloading sample image from: {image_url}")
        original_image = Image.open(requests.get(image_url, stream=True).raw).convert(
            "RGB"
        )
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return

    # ===================================================================
    # STEP 1: Get a Factual Base Caption
    # ===================================================================
    logger.info("\n--- Step 1: Generating base caption ---")
    try:
        img_processor = BlipProcessor.from_pretrained(IMG2TEXT_MODEL)
        img2text_model = BlipForConditionalGeneration.from_pretrained(
            IMG2TEXT_MODEL
        ).to(DEVICE)
    except Exception as e:
        logger.error(f"Could not load vision model: {e}")
        return

    inputs = img_processor(original_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = img2text_model.generate(**inputs, max_length=50)
    base_caption = img_processor.decode(out[0], skip_special_tokens=True)
    logger.info(f"Base caption generated: '{base_caption}'")

    # ===================================================================
    # STEP 2: Elaborate with a Powerful LLM
    # ===================================================================
    logger.info("\n--- Step 2: Elaborating caption into a long description ---")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        # For large models, you may need to load in 4-bit for memory efficiency
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,
            load_in_4bit=True,  # Set to False if you have enough VRAM and no bitsandbytes
        ).to(DEVICE)
    except Exception as e:
        logger.error(
            f"Could not load LLM. Ensure you have 'bitsandbytes' installed if using 4-bit. Error: {e}"
        )
        return

    # Create a detailed prompt for the LLM
    prompt_template = f"""
    [INST] You are a professional, descriptive novelist. Your task is to take a simple image caption and expand it into a vivid, detailed, and elaborate 500-word description. Describe the scene, the atmosphere, the textures, the lighting, potential sounds, and the emotional context. Do not just list items; paint a full picture with your words.

    Simple Caption: "{base_caption}"

    Now, provide the long, detailed description. [/INST]
    """

    messages = [{"role": "user", "content": prompt_template}]

    # Tokenize the chat and immediately move the resulting tensor to the correct device
    tokenized_chat = llm_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    logger.info("Generating long description (this may take a moment)...")
    with torch.no_grad():
        # Ensure the tensor you pass here is the one that was moved to the device
        outputs = llm_model.generate(
            tokenized_chat,
            max_new_tokens=512,
            pad_token_id=llm_tokenizer.eos_token_id,
        ).to(DEVICE)

    # Decode the output, removing the prompt part
    long_description = llm_tokenizer.decode(
        outputs[0][tokenized_chat.shape[-1] :], skip_special_tokens=True
    ).strip()

    # Save the long description
    text_filename = "long_description_representation.txt"
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(long_description)
    logger.info(f"✅ Long description saved to '{text_filename}'")

    # ===================================================================
    # STEP 3: Decompress Text back to Image and Plot
    # ===================================================================
    logger.info("\n--- Step 3: Decompressing text to image and plotting ---")
    try:
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            TEXT2IMG_MODEL, torch_dtype=torch.float16
        ).to(DEVICE)
    except Exception as e:
        logger.error(f"Could not load diffusion model: {e}")
        return

    # Generate image from the long description
    reconstructed_image = text2img_pipe(long_description).images[0]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image", fontsize=14)
    axes[1].axis("off")

    # Save the final comparison plot
    plot_filename = "long_description_comparison.png"
    plt.savefig(plot_filename)
    logger.info(f"✅ Comparison plot saved to '{plot_filename}'")
    logger.info("\n--- Test Finished ---")


if __name__ == "__main__":
    main()
