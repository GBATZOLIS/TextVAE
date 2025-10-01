import torch
import argparse
from torchvision.utils import save_image

from diffusion_model import ConditionalDiffusionModel


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the attributes for the images you want to generate
    # This is a batch of 4 sample conditions
    conditions = torch.tensor(
        [
            # [floor, wall, obj, scale, orient], [sq, cyl, sph, dsp]
            [-1.0, 1.0, 0.5, 0.8, 0.0]
            + [0, 0, 1, 0],  # Blue floor, green wall, pink sphere
            [0.2, -0.8, -1.0, -1.0, 1.0]
            + [1, 0, 0, 0],  # Yellow floor, red wall, blue square, small
            [0.5, 0.5, 0.5, 0.0, -1.0] + [0, 1, 0, 0],  # Grey scene, cylinder
            [1.0, 1.0, -1.0, 1.0, 0.5]
            + [0, 0, 0, 1],  # Green floor/wall, blue dispenser, large
        ],
        dtype=torch.float,
    ).to(device)

    num_samples = conditions.shape[0]
    condition_dim = conditions.shape[1]

    # --- Load Model ---
    model = ConditionalDiffusionModel(
        condition_dim=condition_dim,
        embed_dim=args.embed_dim,
        image_size=args.image_size,
    ).to(device)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # --- Generate Images ---
    print(f"Generating {num_samples} images...")
    with torch.no_grad():
        generated_images = model.sample(num_samples, conditions)

    # De-normalize from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) * 0.5

    save_image(generated_images, args.output_path, nrow=num_samples)
    print(f"Images saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using a trained diffusion model."
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image size the model was trained on.",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Embedding dimension used during training.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="generated_output.png",
        help="Path to save the output image grid.",
    )
    args = parser.parse_args()
    main(args)
