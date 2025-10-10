import torch
from torchvision.utils import make_grid
import wandb


@torch.no_grad()
def save_reconstruction_sample(
    model, image_batch, epoch, device, config, prefix="train"
):
    """
    Saves a grid of original vs. reconstructed images and logs to wandb.
    """
    model.eval()

    # Generate reconstructions using the full, untruncated sequence
    output = model(image_batch, seq_len=None)
    reconstructions = output["reconstructions"]

    # De-normalize images for visualization
    originals_denorm = image_batch * 0.5 + 0.5
    recons_denorm = reconstructions * 0.5 + 0.5

    # We'll show 16 originals and their reconstructions
    num_images_to_save = min(16, image_batch.size(0))

    # Combine the batches
    comparison_batch = torch.cat(
        [originals_denorm[:num_images_to_save], recons_denorm[:num_images_to_save]]
    )

    # --- CORRECTED: Create a single image grid from the batch ---
    # This converts the [32, 3, 32, 32] tensor into a single image tensor
    # that wandb can display. The top row will be originals, bottom row reconstructions.
    grid = make_grid(comparison_batch, nrow=num_images_to_save)

    # Log the image grid to wandb
    wandb.log({f"{prefix}/reconstructions": wandb.Image(grid)})
    print(f"Logged reconstruction sample for epoch {epoch} to wandb.")
