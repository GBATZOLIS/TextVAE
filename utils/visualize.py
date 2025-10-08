# utils/visualize.py

import torch
from torchvision.utils import save_image, make_grid
import os
import wandb


@torch.no_grad()
def save_reconstruction_sample(model, images, epoch, device, config, prefix="train"):
    model.eval()

    n_generate = config.MAX_N_GENERATE // 2

    pixels_quantized = ((images * 0.5 + 0.5) * 255).long()
    pixels_flat = pixels_quantized.permute(0, 2, 3, 1).reshape(images.size(0), -1)
    pixel_input = pixels_flat[:, :-1]

    model_output = model(images, pixel_input, n_generate)
    logits = model_output["pixel_logits"]

    recon_pixels_flat = torch.argmax(logits, dim=-1)

    first_pixel = pixels_flat[:, 0].unsqueeze(1)
    recon_pixels_flat = torch.cat([first_pixel, recon_pixels_flat], dim=1)

    # --- FIX ---
    # The original line incorrectly reshaped the flat pixel tensor, causing a dimension mismatch.
    # The fix is to explicitly reshape to (B, H, W, C) and then permute to (B, C, H, W).
    b, _, h, w = images.shape
    c = 3  # for RGB images
    recon_images = recon_pixels_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    recon_images = recon_images.float() / 255.0
    images_denorm = images * 0.5 + 0.5

    # This line should now work correctly
    comparison = torch.cat([images_denorm[:8], recon_images[:8]])
    grid = make_grid(comparison)

    save_dir = os.path.join(config.RESULTS_DIR, prefix)
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"reconstruction_epoch_{epoch}.png")
    save_image(grid, filepath)

    wandb.log(
        {f"{prefix.capitalize()} Reconstructions Epoch {epoch}": wandb.Image(filepath)}
    )

    print(f"Saved {prefix} reconstruction sample to {filepath}")
