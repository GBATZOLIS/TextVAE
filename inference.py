# inference.py

"""
Script for testing the trained Swin-based Text-Conditioned VAE model.
"""
import argparse
import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from typing import cast

# We need to import the config class so torch.load knows about it
from config import VAEConfig
from models.vae import TextVAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Inference script for the Swin-VAE.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file).",
    )
    p.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./results_swin",
        help="Directory to save the output.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Number of DDIM inference steps for reconstruction.",
    )
    return p.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> TextVAE:
    """Loads the model and the trained weights from a checkpoint."""
    checkpoint_path_expanded = Path(checkpoint_path).expanduser()
    try:
        # FIX: Set weights_only=False to allow loading checkpoints that contain
        # non-tensor objects like the VAEConfig class.
        checkpoint = torch.load(
            checkpoint_path_expanded, map_location=device, weights_only=False
        )

        if "config" in checkpoint:
            cfg = checkpoint["config"]
            logger.info("Configuration loaded from checkpoint.")
        else:
            logger.warning(
                "Checkpoint does not contain config. Initializing with default VAEConfig."
            )
            cfg = VAEConfig()

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path_expanded}.")
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint. Error: {e}")
        raise

    model = TextVAE(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Successfully loaded model weights from {checkpoint_path_expanded}")
    model.eval()
    return cast(TextVAE, model)


def main():
    """Main function to run the inference process."""
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model_from_checkpoint(args.checkpoint, device)
    except Exception:
        logger.error("Could not initialize model from checkpoint. Exiting.")
        return

    image_path_expanded = Path(args.image_path).expanduser()
    try:
        input_image = Image.open(image_path_expanded).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Input image not found at {image_path_expanded}")
        return

    transform = T.Compose(
        [
            T.Resize((model.cfg.image_size, model.cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    logger.info("1. Compressing image to text code...")
    with torch.no_grad():
        img_feat_raw = model.vision_enc_model(image_tensor).last_hidden_state
        img_feat = model.vision_feat_proj(img_feat_raw)

        hard_tokens = torch.full(
            (1, 1), model.start_token_id, dtype=torch.long, device=device
        )
        for _ in range(model.cfg.max_text_length - 1):
            logits, _ = model.text_dec(hard_tokens, img_feat)
            next_token_logits = logits[:, -1, :]
            hard_next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            hard_tokens = torch.cat([hard_tokens, hard_next_token], dim=1)

    generated_text = model.tokenizer.decode(hard_tokens[0], skip_special_tokens=True)
    logger.info(f"   -> Generated Text: '{generated_text}'")

    logger.info(f"2. Decompressing text code to image using {args.steps} steps...")
    with torch.no_grad():
        reconstructed_image_tensor = model.diffusion_dec.sample(
            hard_tokens,
            shape=(
                1,
                model.cfg.image_channels,
                model.cfg.image_size,
                model.cfg.image_size,
            ),
        )

    reconstructed_image_tensor = (reconstructed_image_tensor.clamp(-1, 1) + 1) / 2.0
    reconstructed_image = T.ToPILImage()(reconstructed_image_tensor[0].cpu())

    input_image.resize(reconstructed_image.size).save(output_dir / "original.png")
    reconstructed_image.save(output_dir / "reconstructed.png")
    (output_dir / "generated_text.txt").write_text(generated_text)

    logger.info(f"Results saved to {output_dir.resolve()}")
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
