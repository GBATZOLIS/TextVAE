import torchvision.utils as vutils
import wandb
import os


def save_generated_images(prior_model, vqvae_model, num_images, device, save_path):
    """
    Generates images using the prior and decodes them with the VQ-VAE decoder.
    """
    print(f"Generating {num_images} new images...")

    # 1. Generate new code sequences from the prior
    generated_codes = prior_model.generate(num_samples=num_images, device=device)

    # 2. Look up the codebook embeddings for these codes
    quantized_features = vqvae_model.quantizer.embedding(generated_codes)

    # 3. Decode the features into images
    reconstructed_images = vqvae_model.decoder(quantized_features)
    reconstructed_images = reconstructed_images.mul(0.5).add(
        0.5
    )  # Denormalize from [-1, 1] to [0, 1]

    # 4. Save and log the images
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(reconstructed_images, save_path, nrow=int(num_images**0.5))
    wandb.log({"generated_images": wandb.Image(save_path)})

    print(f"Saved generated images to {save_path}")
