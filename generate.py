# generate.py
# NOTE: This script is a placeholder for a true autoregressive generation.
# A full implementation would involve generating pixels one by one, which is slow.
# This script demonstrates how to load the model and perform a forward pass for reconstruction.

import torch
import config
from models.vqvae import VQVAE_AR
from utils.visualize import save_reconstruction_sample
from utils.helper import get_data_loaders
import os


def generate():
    device = torch.device(config.DEVICE)
    model_path = "model.pth"  # <-- CHANGE THIS to your trained model path

    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found.")
        print("Please train a model first using 'main.py' and update the path.")
        return

    # --- Load Model ---
    print("Loading model...")
    model = VQVAE_AR(
        vq_vae_config=config.VQ_VAE_CONFIG,
        ar_config=config.AUTOREGRESSIVE_CONFIG,
        decoder_config=config.DECODER_CONFIG,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.")

    # --- Get Data Sample ---
    print("Loading data sample...")
    data_loader = get_data_loaders(config.DATASET_PATH, config.BATCH_SIZE)
    sample_images, _ = next(iter(data_loader))
    sample_images = sample_images.to(device)

    # --- Generate Reconstruction ---
    print("Generating reconstruction...")
    # This function is repurposed here to show a model forward pass
    # For true generation, you'd build a new pixel-by-pixel generation loop
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_reconstruction_sample(model, sample_images, "final", device, config)
    print(f"Reconstruction saved in '{config.RESULTS_DIR}' directory.")


if __name__ == "__main__":
    generate()
