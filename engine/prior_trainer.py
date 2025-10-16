import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import config


class PriorTrainer:
    """
    Trainer for the autoregressive Prior model.
    """

    def __init__(
        self, prior_model, vqvae_model, train_loader, val_loader, optimizer, device
    ):
        self.prior_model = prior_model
        self.vqvae_model = vqvae_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Freeze the VQ-VAE model as it's only used for encoding
        self.vqvae_model.eval()
        for param in self.vqvae_model.parameters():
            param.requires_grad = False

    def train_epoch(self, epoch):
        self.prior_model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training Prior]")

        for images, _ in progress_bar:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            # 1. Get discrete codes from the pre-trained VQ-VAE
            with torch.no_grad():
                encoded_features = self.vqvae_model.encoder(
                    self.vqvae_model.patch_embedding(images)
                )
                indices = self.vqvae_model.quantizer.get_code_indices(encoded_features)

            # 2. Prepare inputs and targets for the Prior
            # Input is the sequence except the last token
            inputs = indices[:, :-1]
            # Target is the sequence except the first token
            targets = indices[:, 1:]

            # 3. Forward pass through the Prior
            logits = self.prior_model(inputs)

            # 4. Calculate loss
            # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            wandb.log({"train/prior_batch_loss": loss.item(), "epoch": epoch})

        avg_loss = total_loss / len(self.train_loader)
        wandb.log({"train/prior_avg_epoch_loss": avg_loss, "epoch": epoch})
        print(f"Training Epoch {epoch} Avg Loss: {avg_loss:.4f}")

    def run(self):
        for epoch in range(1, config.PRIOR_NUM_EPOCHS + 1):
            self.train_epoch(epoch)
            # Note: A validation loop could be added here for more robust training

            # Save the model at the end of training
            if epoch == config.PRIOR_NUM_EPOCHS:
                torch.save(self.prior_model.state_dict(), config.PRIOR_CHECKPOINT_PATH)
                print(f"Prior model saved to {config.PRIOR_CHECKPOINT_PATH}")
