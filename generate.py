import torch
from torchvision.utils import save_image
import os

import config
from models.vqvae import VQVAE


@torch.no_grad()
def generate_samples(model, num_samples, device):
    """
    Generates new image samples from the direct reconstruction VQ-VAE decoder.

    NOTE: This function uses a naive, random sampling of latent codes. This will
    result in noisy, incoherent images because the decoder expects a spatially
    structured sequence of codes, not random ones. For high-quality generation,
    these codes should be generated from a trained prior model.
    """
    model.eval()

    # 1. Create a batch of random latent codes.
    # These are random indices into the codebook.
    random_indices = torch.randint(
        low=0,
        high=config.NUM_EMBEDDINGS,
        size=(num_samples, config.NUM_PATCHES),
        device=device,
    )

    # 2. Look up the corresponding embedding vectors from the codebook.
    quantized_features = model.quantizer.embedding(random_indices)

    # 3. Decode the vectors in a single forward pass to generate images.
    # The decoder takes the full sequence of vectors and reconstructs an image.
    generated_images = model.decoder(quantized_features)

    # De-normalize the generated images from [-1, 1] to [0, 1] for saving.
    generated_images_denorm = generated_images * 0.5 + 0.5

    return generated_images_denorm


def main():
    """
    Main function to load the trained model and generate new samples.
    """
    # --- Configuration ---
    # NOTE: You must provide a path to a trained model checkpoint
    MODEL_PATH = "/home/rg625/mnt/TextVAE/wandb/run-20251009_171927-crbijk7s/files/patch_4/checkpoints/model_epoch_2000.pth"
    NUM_SAMPLES = 64
    OUTPUT_DIR = "generated"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Using device: {config.DEVICE}")

    # --- Load Model ---
    model = VQVAE().to(config.DEVICE)
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=config.DEVICE), strict=True
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print(
            "Please train the model and update the MODEL_PATH variable in this script."
        )
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- Generate and Save Samples ---
    print(f"Generating {NUM_SAMPLES} new samples...")
    samples = generate_samples(model, NUM_SAMPLES, config.DEVICE)

    # Save the grid of generated images
    save_path = os.path.join(OUTPUT_DIR, "generated_samples_reconstruction.png")
    save_image(samples, save_path, nrow=int(NUM_SAMPLES**0.5))
    print(f"Saved generated samples to '{save_path}'")
    print(
        "NOTE: The images will look noisy because a proper prior model was not used to generate the latent codes."
    )


if __name__ == "__main__":
    main()
