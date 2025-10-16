# main.py

import torch
import torch.optim as optim
import wandb
import config
from data.dataset import get_cifar10_loaders
from models.vqvae import VQVAE
from engine.trainer import VQVAETrainer


def main():
    """
    Main function to set up and run the training process for the VQ-VAE.
    """
    # Create a dictionary of hyperparameters for wandb logging
    hyperparameters = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("__")
        and isinstance(getattr(config, key), (int, float, str, bool))
    }

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="vision-transformer-vqvae", config=hyperparameters)

    # --- Data Loading ---
    train_loader, val_loader = get_cifar10_loaders(
        dataset_path=config.DATASET_PATH, batch_size=config.VQVAE_BATCH_SIZE
    )

    # --- Model, Optimizer, and Scheduler ---
    model = VQVAE().to(config.DEVICE)
    print(
        f"VQ-VAE model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.VQVAE_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --- NEW: Add a learning rate scheduler for stable convergence ---
    # This will gradually decrease the learning rate, preventing instability in later epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.VQVAE_NUM_EPOCHS,
        eta_min=1e-6,  # Decays to a very small LR
    )

    # --- K-Means Initialization of the Codebook ---
    if not model.quantizer._initialized.item():
        print("Performing one-time K-Means initialization...")
        init_images, _ = next(iter(train_loader))
        init_images = init_images.to(config.DEVICE)
        with torch.no_grad():
            patches = model.patch_embedding(init_images)
            encoded_features = model.encoder(patches)
        model.quantizer.k_means_init(encoded_features)

    # --- Training ---
    # Pass the scheduler to the trainer
    trainer = VQVAETrainer(
        model, train_loader, val_loader, optimizer, scheduler, config.DEVICE, config
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
