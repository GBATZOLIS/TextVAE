# test_off_the_shelf.py

"""
Standalone script to test the off-the-shelf performance of the core
pre-trained models (LLM Prior and Diffusion Decoder concept) before
fine-tuning within the TextVAE framework.
"""
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
from typing import cast

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_llm_prior(model_name: str = "gpt2"):
    """
    Tests the LLM prior by checking if it assigns higher probability to
    a coherent sentence than a jumbled one.

    Args:
        model_name (str): The name of the pre-trained GPT-2 model to load.
    """
    logger.info("--- 1. Testing LLM Prior ---")
    logger.info(f"Loading {model_name} model and tokenizer...")

    # Load pre-trained model and tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        logger.error(
            f"Failed to load '{model_name}'. Check model name and internet connection: {e}"
        )
        return

    # Define test sentences
    coherent_text = "a man with blond hair and a gray hat"
    jumbled_text = "hat gray a with man blond a hair"

    logger.info(f"Coherent text: '{coherent_text}'")
    logger.info(f"Jumbled text:   '{jumbled_text}'")

    def get_text_log_prob(text: str) -> float:
        """Calculates the total log probability of a given text sequence."""
        with torch.no_grad():
            # Tokenize and get input IDs
            token_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

            # Get logits from the model
            outputs = model(token_ids, labels=token_ids)

            # The cross-entropy loss is the negative log-likelihood.
            # We negate it to get the log probability.
            # Multiply by sequence length to get total log prob, not average.
            log_prob = -outputs.loss.item() * (token_ids.shape[1] - 1)

        return cast(float, log_prob)

    # Calculate and compare log probabilities
    coherent_prob = get_text_log_prob(coherent_text)
    jumbled_prob = get_text_log_prob(jumbled_text)

    logger.info(f"Log probability of coherent text: {coherent_prob:.4f}")
    logger.info(f"Log probability of jumbled text:   {jumbled_prob:.4f}")

    if coherent_prob > jumbled_prob:
        logger.info(
            "✅ SUCCESS: Coherent text is correctly identified as more probable."
        )
    else:
        logger.error("❌ FAILED: Jumbled text was incorrectly seen as more probable.")

    logger.info("-" * 28)


def test_diffusion_decoder(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """
    Tests the core text-to-image capability of the diffusion decoder concept
    by generating an image from a text prompt using a standard pre-trained
    Stable Diffusion pipeline.

    Args:
        model_id (str): The Hugging Face model ID for the diffusion pipeline.
    """
    logger.info("\n--- 2. Testing Diffusion Decoder Concept ---")
    logger.info(f"Loading pre-trained diffusion pipeline: '{model_id}'...")

    # Define a descriptive prompt
    prompt = "a man with blond hair and a gray hat, photorealistic, 4k"

    try:
        # Load the pre-trained text-to-image pipeline
        # This is the standard "off-the-shelf" way to use a diffusion model
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(DEVICE)
    except Exception as e:
        logger.error(
            f"Failed to load '{model_id}'. Check model name and internet connection: {e}"
        )
        return

    logger.info(f"Generating image with prompt: '{prompt}'")

    # Generate the image
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.autocast("cuda"):
                result_image = pipe(prompt).images[0]
        else:
            result_image = pipe(prompt).images[0]

    # Save the image to a file
    output_filename = "test_diffusion_output.png"
    result_image.save(output_filename)

    logger.info(f"✅ SUCCESS: Image generated and saved to '{output_filename}'.")
    logger.info("Please visually inspect the image to confirm it matches the prompt.")
    logger.info("-" * 43)


def main():
    """Runs all off-the-shelf tests."""
    test_llm_prior()
    test_diffusion_decoder()


if __name__ == "__main__":
    main()
