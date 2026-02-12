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

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        total_steps = len(train_loader) * config.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )

        if config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))

    def unnormalize_image(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)

    def generate_controlled(
        self, image, target_lens, temperature=1.0, top_k=50, repetition_penalty=1.2
    ):
        self.model.eval()
        B = image.shape[0]
        device = self.device

        input_ids = torch.full(
            (B, 1), self.eos_token_id, dtype=torch.long, device=device
        )
        target_len_tensor = target_lens.to(device)

        current_lengths = torch.zeros(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        while not finished.all():
            with torch.no_grad():
                logits = self.model(image, input_ids, target_len_tensor)
                next_token_logits = logits[:, -1, :]

                # --- Repetition Penalty ---
                # Penalize tokens that have already been generated to prevent loops
                if repetition_penalty != 1.0:
                    for i in range(B):
                        # Get unique tokens generated so far
                        unique_tokens = torch.unique(input_ids[i])

                        # Exclude EOS token from penalty so we don't discourage stopping
                        # (Since SOS is also EOS ID, and we NEED the model to predict EOS at the end)
                        mask = unique_tokens != self.eos_token_id
                        unique_tokens = unique_tokens[mask]

                        if len(unique_tokens) > 0:
                            # Apply penalty to these tokens
                            # If logit < 0 (unlikely), multiply by penalty to make it smaller (more negative)
                            # If logit > 0 (likely), divide by penalty to make it smaller (less positive)
                            selected_logits = next_token_logits[i, unique_tokens]
                            next_token_logits[i, unique_tokens] = torch.where(
                                selected_logits < 0,
                                selected_logits * repetition_penalty,
                                selected_logits / repetition_penalty,
                            )

                # Sampling logic
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_k_probs, _ = torch.topk(next_token_logits, top_k)
                    min_val = top_k_probs[:, -1].unsqueeze(-1)
                    next_token_logits[next_token_logits < min_val] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Pad with EOS if finished
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

        generated_ids = self.generate_controlled(images, lengths)
        vis_images = self.unnormalize_image(images)
        wandb_images = []

        for i in range(len(images)):
            target_k = lengths[i].item()

            full_gen = generated_ids[i].tolist()

            # 1. Find the ACTUAL stopping point
            try:
                first_eos_index = full_gen.index(self.eos_token_id)
                actual_length = first_eos_index + 1
            except ValueError:
                actual_length = -1

            # 2. Decode text
            display_slice = (
                full_gen[:actual_length] if actual_length != -1 else full_gen
            )
            text = self.tokenizer.decode(display_slice)
            clean_text = text.replace("<|endoftext|>", "[EOS]")

            # 3. Create Token Breakdown View
            # This shows exactly what the model counted
            tokens_readable = [
                f"'{self.tokenizer.decode([t])}'" if t != self.eos_token_id else "[EOS]"
                for t in display_slice
            ]

            # Instead of iterating over inp_ids (full batch)
            # iterate over the current sample sequence
            tokens_gt_readable = [
                (
                    f"'{self.tokenizer.decode([gt])}'"
                    if gt != self.eos_token_id
                    else "[EOS]"
                )
                for gt in inp_ids[i].tolist()[
                    :target_k
                ]  # Use the i-th sample and trim to target length
            ]

            if actual_length == target_k:
                status = "SUCCESS: Exact Length Match"
            elif actual_length == -1:
                status = "FAIL: No EOS generated"
            else:
                status = f"FAIL: Stopped at {actual_length} (Goal: {target_k})"

            gt_seq = inp_ids[i].tolist()[:target_k]
            gt_text = self.tokenizer.decode(gt_seq).replace("<|endoftext|>", "[EOS]")

            caption = (
                f"Goal: {target_k} tokens\n"
                f"Status: {status}\n\n"
                f"Prediction ({len(tokens_readable)} toks):\n{clean_text}\n"
                f"Breakdown: {tokens_readable}\n\n"
                f"Ground Truth:\n{gt_text}\n"
                f"Ground Truth Breakdown: {tokens_gt_readable}\n"
            )

            wandb_images.append(wandb.Image(vis_images[i], caption=caption))

        wandb.log({"Validation Samples": wandb_images, "epoch": epoch})

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, (images, inp_ids, labels, lengths) in enumerate(pbar):
            images = images.to(self.device)
            inp_ids = inp_ids.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

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
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(
                {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
            )

            if self.config.use_wandb:
                wandb.log(
                    {"train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                )

        self.log_predictions(epoch)
        return total_loss / len(self.train_loader)

    def generate_with_custom_length(
        self,
        target_length: int,
        num_samples: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """
        Generate captions from validation images using a user-specified target length.
        """

        self.model.eval()

        if self.val_loader is None:
            raise ValueError("Validation loader not available.")

        # Grab one batch
        images, _, _, _ = next(iter(self.val_loader))

        images = images[:num_samples].to(self.device)

        # 🔥 Here is the important part:
        target_lengths = torch.full(
            (images.size(0),), target_length, dtype=torch.long, device=self.device
        )

        generated_ids = self.generate_controlled(
            images, target_lengths, temperature=temperature, top_k=top_k
        )

        results = []

        for i in range(images.size(0)):
            tokens = generated_ids[i].tolist()

            if self.eos_token_id in tokens:
                eos_idx = tokens.index(self.eos_token_id)
                tokens = tokens[: eos_idx + 1]

            text = self.tokenizer.decode(tokens).replace("<|endoftext|>", "")

            print(f"\nTarget Length: {target_length}")
            print(f"Actual Length: {len(tokens)}")
            print(text)

            results.append(
                {
                    "target_length": target_length,
                    "actual_length": len(tokens),
                    "text": text,
                }
            )

        return results

    def save(self, path, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"] + 1
