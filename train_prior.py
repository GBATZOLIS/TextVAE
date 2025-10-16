import torch
import wandb
import config
from data.dataset import get_cifar10_loaders
from models.vqvae import VQVAE
from models.prior import PriorTransformer
from engine.prior_trainer import PriorTrainer


def main():
    """
    Main function to set up and run the training process for the Prior model.
    """
    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="vq-vae-prior-generation", config=config)

    # --- Data Loading ---
    train_loader, val_loader = get_cifar10_loaders(
        dataset_path=config.DATASET_PATH, batch_size=config.PRIOR_BATCH_SIZE
    )

    # --- Load Pre-trained VQ-VAE ---
    print(f"Loading pre-trained VQ-VAE from {config.VQVAE_CHECKPOINT_PATH}...")
    try:
        vqvae_model = VQVAE()
        vqvae_model.load_state_dict(
            torch.load(config.VQVAE_CHECKPOINT_PATH, map_location=config.DEVICE)
        )
        vqvae_model.to(config.DEVICE)
        print("VQ-VAE model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Error: VQ-VAE checkpoint not found at '{config.VQVAE_CHECKPOINT_PATH}'."
        )
        print("Please train the VQ-VAE first or update the path in config.py.")
        return

    # --- Instantiate Prior Model and Optimizer ---
    prior_model = PriorTransformer().to(config.DEVICE)
    print(
        f"Prior model has {sum(p.numel() for p in prior_model.parameters() if p.requires_grad):,} trainable parameters."
    )
    optimizer = torch.optim.AdamW(
        prior_model.parameters(), lr=config.PRIOR_LEARNING_RATE
    )

    # --- Training ---
    trainer = PriorTrainer(
        prior_model, vqvae_model, train_loader, val_loader, optimizer, config.DEVICE
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
