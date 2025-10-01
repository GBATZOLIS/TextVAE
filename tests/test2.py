# test_compression_loop_with_plot.py

"""
Simulates the full image -> text -> image compression loop and generates
a side-by-side comparison plot.

1.  ENCODING (Compression):   An image is "compressed" into a text
                               description and saved to a .txt file.
2.  DECODING (Decompression): The text description is "decompressed"
                               back into an image.
3.  PLOTTING:                 The original and reconstructed images are
                               plotted side-by-side for direct comparison.
"""
import torch
import requests
from PIL import Image
import logging
import textwrap

# Import the necessary Hugging Face libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# Import for plotting
import matplotlib.pyplot as plt

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model IDs ---
# Encoder Model: A powerful image-to-text model.
IMG2TEXT_MODEL = "Salesforce/blip-image-captioning-large"
# Decoder Model: A standard, high-quality text-to-image model.
TEXT2IMG_MODEL = "runwayml/stable-diffusion-v1-5"


def main():
    """
    Executes the full compression and decompression test loop.
    """
    # --- Load a sample image from a URL ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        logger.info(f"Downloading sample image from: {image_url}")
        original_image = Image.open(requests.get(image_url, stream=True).raw).convert(
            "RGB"
        )
    except Exception as e:
        logger.error(f"Failed to download or process the image: {e}")
        return

    # ===================================================================
    # STEP 1: ENCODING (Image -> Text Compression)
    # ===================================================================
    logger.info("\n--- Starting Step 1: Image to Text (Compression) ---")

    try:
        logger.info(f"Loading Image-to-Text model: {IMG2TEXT_MODEL}")
        img_processor = BlipProcessor.from_pretrained(IMG2TEXT_MODEL)
        img2text_model = BlipForConditionalGeneration.from_pretrained(
            IMG2TEXT_MODEL
        ).to(DEVICE)
    except Exception as e:
        logger.error(f"Failed to load the image captioning model: {e}")
        return

    inputs = img_processor(original_image, return_tensors="pt").to(DEVICE)
    logger.info("Compressing image into text...")
    with torch.no_grad():
        output_ids = img2text_model.generate(**inputs, max_length=250, num_beams=8)
    compressed_text = img_processor.decode(output_ids[0], skip_special_tokens=True)

    # --- NEW: Save the compressed text to a file ---
    text_filename = "compressed_representation.txt"
    with open(text_filename, "w") as f:
        f.write(compressed_text)

    logger.info("✅ Compression complete!")
    logger.info(f"   - Text representation saved to '{text_filename}'")

    # ===================================================================
    # STEP 2: DECODING (Text -> Image Decompression)
    # ===================================================================
    logger.info("\n--- Starting Step 2: Text to Image (Decompression) ---")

    try:
        logger.info(f"Loading Text-to-Image model: {TEXT2IMG_MODEL}")
        text2img_pipe = StableDiffusionPipeline.from_pretrained(TEXT2IMG_MODEL).to(
            DEVICE
        )
    except Exception as e:
        logger.error(f"Failed to load the diffusion model: {e}")
        return

    prompt = compressed_text
    logger.info("Decompressing text back into an image...")
    with torch.no_grad():
        # Use automatic mixed precision for faster inference on GPU
        autocast_enabled = DEVICE == "cuda"
        with torch.autocast(DEVICE, enabled=autocast_enabled):
            reconstructed_image = text2img_pipe(prompt).images[0]

    logger.info("✅ Decompression complete!")

    # ===================================================================
    # STEP 3: PLOT AND COMPARE
    # ===================================================================
    logger.info("\n--- Starting Step 3: Plotting Comparison ---")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Plot Reconstructed Image
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image", fontsize=14)
    axes[1].axis("off")

    # Add a title to the whole plot with the compressed text
    wrapped_text = "\n".join(
        textwrap.wrap(f"Compressed Text: '{compressed_text}'", width=100)
    )
    fig.suptitle(wrapped_text, fontsize=10, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])

    # Save the plot
    plot_filename = "comparison_plot.png"
    plt.savefig(plot_filename)

    logger.info(f"✅ Comparison plot saved to '{plot_filename}'")
    logger.info("\n--- Test Finished ---")


if __name__ == "__main__":
    main()
