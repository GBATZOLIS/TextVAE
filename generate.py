import torch
import config
from models.vqvae import VQVAE
from models.prior import PriorTransformer
from utils.visualize import save_generated_images
import wandb


def main():
    """
    Loads trained models and generates new images by sampling from the prior.
    """
    # Use a dummy wandb run for logging the generated image artifact
    wandb.init(project="vq-vae-prior-generation", job_type="generation")

    # --- Load Pre-trained VQ-VAE ---
    print(f"Loading pre-trained VQ-VAE from {config.VQVAE_CHECKPOINT_PATH}...")
    try:
        vqvae_model = VQVAE()
        vqvae_model.load_state_dict(
            torch.load(config.VQVAE_CHECKPOINT_PATH, map_location=config.DEVICE)
        )
        vqvae_model.to(config.DEVICE).eval()
    except FileNotFoundError:
        print(
            f"Error: VQ-VAE checkpoint not found at '{config.VQVAE_CHECKPOINT_PATH}'. Cannot proceed."
        )
        return

    # --- Load Trained Prior Model ---
    print(f"Loading trained Prior from {config.PRIOR_CHECKPOINT_PATH}...")
    try:
        prior_model = PriorTransformer()
        prior_model.load_state_dict(
            torch.load(config.PRIOR_CHECKPOINT_PATH, map_location=config.DEVICE)
        )
        prior_model.to(config.DEVICE).eval()
    except FileNotFoundError:
        print(
            f"Error: Prior checkpoint not found at '{config.PRIOR_CHECKPOINT_PATH}'. Cannot proceed."
        )
        return

    # --- Generate and Save Images ---
    save_generated_images(
        prior_model=prior_model,
        vqvae_model=vqvae_model,
        num_images=64,  # Generate an 8x8 grid
        device=config.DEVICE,
        save_path="outputs/generated_samples.png",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
