import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import wandb
from torchvision.utils import save_image, make_grid

# UPDATED: Import from the new unified dataset and custom diffusion model files
from shapes3d import Shapes3DDataset
from conditional_diffusion import ConditionalDiffusionModel


def main(args):
    wandb.init(project=args.project_name, config=args)
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset and DataLoader ---
    dataset = Shapes3DDataset(image_size_diff=config.image_size, split="train")
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )

    condition_dim = 5 + dataset.NUM_SHAPES

    # --- Model and Optimizer ---
    model = ConditionalDiffusionModel(
        condition_dim=condition_dim,
        embed_dim=config.embed_dim,
        image_size=config.image_size,  # Now 224
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Training Loop ---
    step = 0
    for epoch in range(config.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            # UPDATED: Use the correct dictionary keys from the unified dataset
            images = batch["image_diff"].to(device)
            conditions = batch["condition"].to(device)

            loss = model(images, conditions)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
            wandb.log({"loss": loss.item(), "step": step})
            step += 1

        # --- Generate and Log Samples ---
        if (epoch + 1) % config.log_every == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                sample_conditions = sample_batch["condition"][: config.num_samples].to(
                    device
                )

                # UPDATED: Pass the device to the custom sample method
                generated_images = model.sample(
                    config.num_samples, sample_conditions, device
                )
                generated_images = (generated_images + 1) * 0.5

                grid = make_grid(generated_images, nrow=4)
                wandb.log({"generated_samples": wandb.Image(grid), "epoch": epoch})
                save_image(grid, f"samples_epoch_{epoch+1}.png")
            model.train()

        # --- Save Checkpoint ---
        if (epoch + 1) % config.save_every == 0:
            torch.save(model.state_dict(), f"diffusion_model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a custom conditional diffusion model."
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Reduce batch size for larger 224x224 images.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # UPDATED: Default image size now matches the language model
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Dimension of the attribute embedding.",
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log sample images every N epochs."
    )
    parser.add_argument(
        "--save_every", type=int, default=50, help="Save a checkpoint every N epochs."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of images to generate for logging (reduce for memory).",
    )
    parser.add_argument(
        "--project_name", type=str, default="CustomDiffusion-224px-Pipeline"
    )
    args = parser.parse_args()
    main(args)
