import torch
import wandb
import config
from data.dataset import get_cifar10_loaders
from models.vqvae import VQVAE
from engine.trainer import Trainer


def main():
    """
    Main function to set up and run the training process.
    """
    # Create a dictionary of hyperparameters for wandb logging
    hyperparameters = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("__")
        and isinstance(getattr(config, key), (int, float, str, bool))
    }

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="vision-transformer-vqvae-reconstruction", config=hyperparameters
    )

    # --- Data Loading ---
    train_loader, val_loader = get_cifar10_loaders(
        dataset_path=config.DATASET_PATH, batch_size=config.BATCH_SIZE, num_workers=4
    )

    # --- Model, Optimizer ---
    model = VQVAE().to(config.DEVICE)
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- NEW: K-Means Initialization of the Codebook ---
    # This is done once before training starts to prevent codebook collapse.
    if not model.quantizer._initialized.item():
        # Get one batch of data for initialization
        init_images, _ = next(iter(train_loader))
        init_images = init_images.to(config.DEVICE)
        # Pass through encoder to get features for clustering
        with torch.no_grad():
            patches = model.patch_embedding(init_images)
            encoded_features = model.encoder(patches)
        # Initialize the codebook with the centroids of the features
        model.quantizer.k_means_init(encoded_features)

    # --- Training ---
    trainer = Trainer(model, train_loader, val_loader, optimizer, config.DEVICE, config)
    trainer.run()


if __name__ == "__main__":
    main()
