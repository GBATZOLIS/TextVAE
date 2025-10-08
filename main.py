# main.py

import torch
import torch.optim as optim
import wandb
import config
import os
from models.vqvae import VQVAE_AR
from engine.trainer import Trainer
from utils.helper import set_seed, get_data_loaders
from utils.logger import logger


def main():
    # --- Setup ---
    set_seed(config.SEED)
    wandb.init(
        project=config.PROJECT_NAME,
        entity=config.WANDB_ENTITY,
        name=config.RUN_NAME,
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "d_model": config.D_MODEL,
            "max_n_generate": config.MAX_N_GENERATE,
        },
    )

    # --- Data ---
    logger.info("Loading data...")
    train_loader, val_loader = get_data_loaders(
        path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        val_batch_size=config.VALIDATION_BATCH_SIZE,
    )

    # --- Model ---
    logger.info("Initializing model...")
    model = VQVAE_AR(
        vq_vae_config=config.VQ_VAE_CONFIG,
        ar_config=config.AUTOREGRESSIVE_CONFIG,
        decoder_config=config.DECODER_CONFIG,
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    wandb.watch(model, log="all", log_freq=500)

    # --- Training ---
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=config.DEVICE,
        config=config,
    )
    trainer.run()

    logger.info("Training finished.")
    # Save final model artifact to wandb
    final_model_path = os.path.join(wandb.run.dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    artifact = wandb.Artifact(config.RUN_NAME, type="model")
    artifact.add_file(final_model_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
