# test.py
import torch
from PIL import Image
import argparse
from pathlib import Path
import os

from config import config
from models.codec import Codec  # Import the single Codec model


def load_image(image_path, size=512):
    """Loads and resizes an image."""
    return Image.open(image_path).convert("RGB").resize((size, size))


def main(args):
    print(f"--- Running Inference with method: {args.method} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the model architecture
    model = Codec(config, device)

    # 2. Load trained weights if a checkpoint is provided
    if args.checkpoint_dir:
        print(f"Loading trained weights from: {args.checkpoint_dir}")
        # Load the LoRA adapter weights into the VLM
        vlm_adapter_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(vlm_adapter_path):
            model.vlm_model.load_adapter(args.checkpoint_dir)
            print("Successfully loaded VLM LoRA adapters.")
        else:
            print(f"Warning: VLM adapter file not found at {vlm_adapter_path}")

        # Load the projection layer weights
        projection_path = os.path.join(args.checkpoint_dir, "projection.bin")
        if os.path.exists(projection_path):
            model.projection.load_state_dict(
                torch.load(projection_path, map_location=device)
            )
            print("Successfully loaded projection layer.")
        else:
            print(f"Warning: Projection layer file not found at {projection_path}")
    else:
        print("Warning: No checkpoint provided. Using pre-trained weights only.")

    model.eval()

    # 3. Encode: Image -> Text
    print(f"\nEncoding image: {args.input_image}")
    image_pil = load_image(args.input_image)
    generated_text = model.encode(image_pil)
    print(f"\n✅ Generated Text:\n{generated_text}")

    # 4. Decode: Text -> Image using the chosen method
    print(f"\nDecoding with '{args.method}' method...")
    if args.method == "long_context":
        reconstructed_image = model.decode_long_context(
            generated_text, seed=config.system.seed
        )
    elif args.method == "controlnet":
        reconstructed_image = model.decode_controlnet(
            generated_text, seed=config.system.seed
        )
    elif args.method == "hybrid":
        reconstructed_image = model.decode_hybrid(
            generated_text, seed=config.system.seed
        )
    else:
        raise ValueError("Invalid method specified.")

    # 5. Save the result
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    final_path = output_path / f"reconstructed_{args.method}.png"
    reconstructed_image.save(final_path)
    print(f"\n✅ Reconstructed image saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a specified decoding method."
    )
    parser.add_argument(
        "method",
        type=str,
        choices=["long_context", "controlnet", "hybrid"],
        help="Decoding method to use.",
    )
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory saved by Accelerate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save the output image.",
    )
    args = parser.parse_args()
    main(args)
