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
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token_id = 50256

        self.optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        # Ignore -100 to prevent learning "EOS after EOS"
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        if config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))

    def unnormalize_image(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)

    def generate_natural(self, image, target_lens):
        """
        Generates exactly target_len tokens per sample.
        Stops generation individually once target is reached.
        """
        self.model.eval()

        B = image.shape[0]
        target_len_tensor = target_lens.to(self.device)

        input_ids = torch.full(
            (B, 1), self.eos_token_id, dtype=torch.long, device=self.device
        )

        # Track how many tokens generated per sample
        current_lengths = torch.zeros(B, dtype=torch.long, device=self.device)

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        while not finished.all():
            with torch.no_grad():
                logits = self.model(image, input_ids, target_len_tensor)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # Only update unfinished sequences
            next_token = torch.where(
                finished, torch.full_like(next_token, self.eos_token_id), next_token
            )

            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

            current_lengths += (~finished).long()
            finished = current_lengths >= target_len_tensor

        return input_ids[:, 1:]

    def log_predictions(self, epoch, num_samples=4):
        if not self.config.use_wandb:
            return

        loader = self.val_loader if self.val_loader else self.train_loader
        try:
            images, inp_ids, _, lengths = next(iter(loader))
        except StopIteration:
            return

        images = images[:num_samples].to(self.device)
        lengths = lengths[:num_samples].to(self.device)

        generated_ids = self.generate_natural(images, lengths)
        vis_images = self.unnormalize_image(images)
        wandb_images = []

        for i in range(len(images)):
            target_k = lengths[i].item()
            pred_seq = generated_ids[i].tolist()

            # Since we didn't force stop in the loop, let's see where the model put EOS
            try:
                # Find first EOS
                actual_stop_index = pred_seq.index(self.eos_token_id) + 1
            except ValueError:
                actual_stop_index = -1  # Never stopped

            # Slice for decoding up to target (or full generated if it failed)
            display_seq = pred_seq[:target_k]
            text = self.tokenizer.decode(display_seq)
            clean_text = text.replace("<|endoftext|>", "[EOS]")
            gt_seq = inp_ids[i].tolist()
            gt_seq = gt_seq[:target_k]  # trim to target length
            gt_text = self.tokenizer.decode(gt_seq)
            gt_clean = gt_text.replace("<|endoftext|>", "[EOS]")

            status = (
                "Correct"
                if actual_stop_index == target_k
                else f"Wrong: Stopped at {actual_stop_index} out of {len(gt_clean)}"
            )

            caption = (
                f"Goal: {target_k} tokens\n"
                f"Result: {status}\n\n"
                f"Prediction:\n{clean_text}\n\n"
                f"Ground Truth:\n{gt_clean}"
            )

            wandb_images.append(wandb.Image(vis_images[i], caption=caption))

        wandb.log({"Validation Samples": wandb_images, "epoch": epoch})

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, (images, inp_ids, labels, lengths) in enumerate(pbar):
            images, inp_ids, labels, lengths = (
                images.to(self.device),
                inp_ids.to(self.device),
                labels.to(self.device),
                lengths.to(self.device),
            )

            # Input Prep: Prepend SOS (using EOS ID)
            B = inp_ids.shape[0]
            sos_token = torch.full(
                (B, 1), self.eos_token_id, dtype=torch.long, device=self.device
            )
            model_inp = torch.cat([sos_token, inp_ids[:, :-1]], dim=1)

            self.optimizer.zero_grad()

            logits = self.model(images, model_inp, lengths)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            if self.config.use_wandb:
                wandb.log({"train_loss": loss.item()})

        self.log_predictions(epoch)
        return total_loss / len(self.train_loader)

    def inference(self, override_length=None):
        """
        Runs inference on validation set.

        Args:
            override_length (int, optional):
                If provided, forces all samples to generate this length.

        Returns:
            List of tuples:
                (prediction_text, ground_truth_text, target_length)
        """
        self.model.eval()
        results = []

        loader = self.val_loader if self.val_loader else self.train_loader

        with torch.no_grad():
            for images, inp_ids, _, lengths in tqdm(loader, desc="Inference"):

                images = images.to(self.device)

                # Optionally override generation length
                if override_length is not None:
                    lengths = torch.full_like(lengths, override_length)
                else:
                    lengths = lengths.to(self.device)

                generated_ids = self.generate_natural(images, lengths)

                B = images.size(0)

                for i in range(B):
                    target_k = lengths[i].item()

                    # ---- Prediction ----
                    pred_seq = generated_ids[i].tolist()[:target_k]
                    pred_text = self.tokenizer.decode(pred_seq)
                    pred_text = pred_text.replace("<|endoftext|>", "[EOS]")

                    # ---- Ground Truth ----
                    gt_seq = inp_ids[i].tolist()[:target_k]
                    gt_text = self.tokenizer.decode(gt_seq)
                    gt_text = gt_text.replace("<|endoftext|>", "[EOS]")

                    results.append((pred_text, gt_text, target_k))

        return results

    def save(self, path, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint["epoch"] + 1  # resume from next epoch
