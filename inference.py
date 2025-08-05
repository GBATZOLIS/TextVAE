# inference.py

"""
Script for testing the trained Text-Conditioned VAE model.

This script performs the following steps:
1.  Loads a trained model checkpoint.
2.  Loads a specified input image.
3.  Encodes the image into its text-based code ("compression").
4.  Decodes the text code back into a reconstructed image ("decompression").
5.  Saves the original image, the generated text, and the reconstructed image
    for qualitative evaluation.

Example Usage:
    python inference.py \
        --checkpoint ./runs/best_model.pt \
        --image_path ./path/to/your/image.jpg \
        --output_dir ./results
"""
import argparse
import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from typing import cast

from config import VAEConfig
from models.vae import TextVAE

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Inference script for the Text-Conditioned VAE."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="./runs/initial_model.pt",
        help="Path to the trained model checkpoint (.pt file).",
    )
    p.add_argument(
        "--image_path",
        type=str,
        default="./results/original.png",
        help="Path to the input image.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the output.",
    )
    p.add_argument(
        "--steps", type=int, default=50, help="Number of DDIM inference steps."
    )
    return p.parse_args()


def load_model_from_checkpoint(
    cfg: VAEConfig, checkpoint_path: str, device: torch.device
) -> TextVAE:
    """Loads the model and the trained weights from a checkpoint."""
    model = TextVAE(cfg).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create a dictionary of the model's trainable components
        model_components = {
            "vision_enc": model.vision_enc,
            "text_dec": model.text_dec,
            "vision_feat_proj": model.vision_feat_proj,
        }

        # Load the state dict for each component if it exists in the checkpoint
        for name, component in model_components.items():
            if name in checkpoint:
                component.load_state_dict(checkpoint[name])
            else:
                logger.warning(
                    f"'{name}' weights not found in checkpoint. Using initialized weights."
                )

        logger.info(f"Successfully loaded model weights from {checkpoint_path}")
    except FileNotFoundError:
        logger.error(
            f"Checkpoint file not found at {checkpoint_path}. Please check the path."
        )
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the checkpoint: {e}")
        raise

    model.eval()
    return cast(TextVAE, model)


def main():
    """Main function to run the inference process."""
    args = parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    cfg = VAEConfig()
    model = load_model_from_checkpoint(cfg, args.checkpoint, device)

    # --- Prepare Input Image ---
    try:
        input_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Input image not found at {args.image_path}")
        return

    transform = T.Compose(
        [
            T.Resize((cfg.image_size, cfg.image_size)),
            T.CenterCrop((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    # --- Run Inference ---
    logger.info("1. Compressing image to text code...")
    with torch.no_grad():
        # Encode the image to get GPT-2 based tokens
        gpt2_tokens = model.encode(image_tensor)["hard_tokens"]

    # Decode GPT-2 tokens to a human-readable text string
    generated_text = model.tokenizer.decode(gpt2_tokens[0], skip_special_tokens=True)
    logger.info(f"   -> Generated Text: '{generated_text}'")

    logger.info("2. Decompressing text code to image...")
    with torch.no_grad():
        # FIX: Bridge the vocabulary gap for the decoder
        # Re-tokenize the generated text using the CLIP tokenizer
        clip_tokens = model.clip_tokenizer(
            generated_text,
            padding="max_length",
            max_length=model.clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Decode using the correctly formatted CLIP tokens
        reconstructed_image_tensor = model.decode(
            clip_tokens, num_inference_steps=args.steps
        )

    # --- Save Results ---
    # De-normalize the image from [-1, 1] to [0, 1] before converting to PIL
    reconstructed_image_tensor = (reconstructed_image_tensor.clamp(-1, 1) + 1) / 2.0
    reconstructed_image = T.ToPILImage()(reconstructed_image_tensor[0].cpu())

    # Save the original and reconstructed images
    input_image.resize(reconstructed_image.size).save(output_dir / "original.png")
    reconstructed_image.save(output_dir / "reconstructed.png")

    # Save the generated text
    with open(output_dir / "generated_text.txt", "w") as f:
        f.write(generated_text)

    logger.info(f"Results saved to {output_dir.resolve()}")
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
