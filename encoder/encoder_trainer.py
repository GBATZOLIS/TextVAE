import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import tiktoken
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader  # Can be None
        self.config = config
        self.device = config.device

        # Tokenizer for decoding logs
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=50256)  # GPT2 PAD/EOS

        if config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))
            # Define metrics
            wandb.define_metric("train_loss", summary="min")

    def unnormalize_image(self, tensor):
        """Reverses ImageNet normalization for visualization"""
        # Mean and Std from dataset.py
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)

        img = tensor * std + mean
        return torch.clamp(img, 0, 1)

    def generate_sample(self, image, target_len):
        """Greedy decoding loop for logging"""
        self.model.eval()
        B = image.shape[0]

        # Start with a dummy token or standard start.
        # Since we trained without explicit SOS, we can start with EOS (50256)
        # or handle cold start in model. We'll assume cold start with dummy EOS.
        input_ids = torch.full((B, 1), 50256, dtype=torch.long, device=self.device)

        target_len_tensor = torch.tensor([target_len] * B, device=self.device)

        for _ in range(target_len):
            with torch.no_grad():
                logits = self.model(image, input_ids, target_len_tensor)
                # Get last token prediction
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids[:, 1:]  # Remove the dummy start token

    def log_predictions(self, epoch, num_samples=4):
        """Logs images and generated captions to WandB"""
        if not self.config.use_wandb:
            return

        # Get a batch
        loader = self.val_loader if self.val_loader else self.train_loader
        images, tokens, lengths = next(iter(loader))

        # Slice
        images = images[:num_samples].to(self.device)
        tokens = tokens[:num_samples].to(self.device)
        lengths = lengths[:num_samples].to(self.device)

        # Generate
        # We use the ground truth length as the target constraint for the planner
        # to see if it can reconstruct the concept GIVEN the correct length.

        # Note: We pick the length of the first sample for simplicity in this greedy loop
        # or we generate individually. Let's do batch generation assuming max length of batch.
        max_len = lengths.max().item()
        generated_ids = self.generate_sample(images, max_len)

        wandb_images = []

        vis_images = self.unnormalize_image(images)

        for i in range(len(images)):
            # Decode Prediction
            pred_text = self.tokenizer.decode(generated_ids[i].tolist())
            # Decode Truth
            true_text = self.tokenizer.decode(tokens[i].tolist())

            # Create Caption
            caption = f"Len: {lengths[i]}\nPred: {pred_text}\nTrue: {true_text}"

            wandb_images.append(wandb.Image(vis_images[i], caption=caption))

        wandb.log({"Validation Samples": wandb_images, "epoch": epoch})
        self.model.train()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, (images, tokens, lengths) in enumerate(pbar):
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            lengths = lengths.to(self.device)

            # Input: [SOS, w1, w2] -> Target: [w1, w2, EOS]
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]

            self.optimizer.zero_grad()

            # Forward
            logits = self.model(images, inp, lengths)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            if self.config.use_wandb:
                wandb.log({"train_loss": loss.item()})

        # Log qualitative results at end of epoch
        self.log_predictions(epoch)

        return total_loss / len(self.train_loader)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
