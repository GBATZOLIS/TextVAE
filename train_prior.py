import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
import os

import config
from data.dataset import get_cifar10_loaders
from models.vqvae import VQVAE
from models.prior import Prior


@torch.no_grad()
def generate_and_log_samples(vqvae_model, prior_model, epoch, device):
    """
    Generates sample images using the LSTM prior and logs them to wandb.
    """
    prior_model.eval()
    vqvae_model.eval()

    num_samples = 64

    generated_codes = prior_model.generate(num_samples, device)

    quantized_features = vqvae_model.quantizer.embedding(generated_codes)
    generated_images = vqvae_model.decoder(quantized_features)
    generated_images_denorm = generated_images * 0.5 + 0.5

    grid = make_grid(generated_images_denorm, nrow=int(num_samples**0.5))
    wandb.log({f"generated_samples_epoch_{epoch}": wandb.Image(grid)})
    print(f"Logged generated samples for epoch {epoch} to wandb.")


def train_prior_epoch(
    vqvae_model, prior_model, dataloader, optimizer, criterion, device
):
    prior_model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training Prior")

    for images, _ in progress_bar:
        images = images.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            patches = vqvae_model.patch_embedding(images)
            encoded_features = vqvae_model.encoder(patches)
            flat_input = encoded_features.view(-1, config.EMBEDDING_DIM)
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(vqvae_model.quantizer.embedding.weight**2, dim=1)
                - 2
                * torch.matmul(flat_input, vqvae_model.quantizer.embedding.weight.t())
            )
            code_indices = torch.argmin(distances, dim=1).view(images.size(0), -1)

        prior_input = code_indices[:, :-1]
        prior_target = code_indices[:, 1:]

        logits = prior_model(prior_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), prior_target.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"train_loss": loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_prior_epoch(vqvae_model, prior_model, dataloader, criterion, device):
    prior_model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating Prior")

    for images, _ in progress_bar:
        images = images.to(device)

        with torch.no_grad():
            patches = vqvae_model.patch_embedding(images)
            encoded_features = vqvae_model.encoder(patches)
            flat_input = encoded_features.view(-1, config.EMBEDDING_DIM)
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(vqvae_model.quantizer.embedding.weight**2, dim=1)
                - 2
                * torch.matmul(flat_input, vqvae_model.quantizer.embedding.weight.t())
            )
            code_indices = torch.argmin(distances, dim=1).view(images.size(0), -1)

        prior_input = code_indices[:, :-1]
        prior_target = code_indices[:, 1:]
        logits = prior_model(prior_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), prior_target.reshape(-1))

        total_loss += loss.item()
        progress_bar.set_postfix({"val_loss": loss.item()})

    return total_loss / len(dataloader)


def main():
    wandb_config = {
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "num_embeddings": config.NUM_EMBEDDINGS,
        "vqvae_embedding_dim": config.EMBEDDING_DIM,
        "patch_size": config.PATCH_SIZE,
        "prior_model": "LSTM",
        "prior_n_blocks": config.PRIOR_N_BLOCKS,
        "prior_embedding_dim": config.PRIOR_EMBEDDING_DIM,
        "prior_dropout": config.PRIOR_DROPOUT,
    }
    wandb.init(project="vqvae-prior-training", config=wandb_config)

    train_loader, val_loader = get_cifar10_loaders(
        dataset_path=config.DATASET_PATH, batch_size=config.BATCH_SIZE, num_workers=4
    )

    print("Loading pre-trained VQ-VAE model...")
    vqvae_model = VQVAE().to(config.DEVICE)
    vqvae_model_path = "/home/rg625/mnt/TextVAE/wandb/run-20251009_171927-crbijk7s/files/patch_4/checkpoints/model_epoch_2000.pth"
    vqvae_model.load_state_dict(
        torch.load(vqvae_model_path, map_location=config.DEVICE)
    )
    vqvae_model.eval()

    # --- CORRECTED: Instantiate Prior with arguments from the config file ---
    prior_model = Prior(
        num_layers=config.PRIOR_N_BLOCKS,
        embedding_dim=config.PRIOR_EMBEDDING_DIM,
        dropout=config.PRIOR_DROPOUT,
    ).to(config.DEVICE)

    print(
        f"Prior model has {sum(p.numel() for p in prior_model.parameters() if p.requires_grad):,} trainable parameters."
    )
    optimizer = torch.optim.AdamW(
        prior_model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print("Starting Prior training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        avg_train_loss = train_prior_epoch(
            vqvae_model, prior_model, train_loader, optimizer, criterion, config.DEVICE
        )
        avg_val_loss = validate_prior_epoch(
            vqvae_model, prior_model, val_loader, criterion, config.DEVICE
        )

        print(
            f"Epoch [{epoch}/{config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        wandb.log(
            {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        if epoch % config.SAVE_IMAGE_INTERVAL == 0:
            generate_and_log_samples(vqvae_model, prior_model, epoch, config.DEVICE)
            model_dir = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"prior_model_epoch_{epoch}.pth")
            torch.save(prior_model.state_dict(), model_path)

    print("Prior training finished.")


if __name__ == "__main__":
    main()
