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

        # --- HPC: Automatic Mixed Precision (AMP) ---
        self.use_amp = getattr(config, "use_amp", False)
        # Use BFloat16 if available (Ampere A100s natively support this perfectly), else Float16
        self.amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        high_lr_params = []
        low_lr_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual_mapper" in name or "countdown_emb" in name:
                high_lr_params.append(param)
            else:
                low_lr_params.append(param)

        base_lr = config.lr

        optimizer_grouped_parameters = [
            {"params": low_lr_params, "lr": base_lr},
            {"params": high_lr_params, "lr": base_lr * 10},
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

        total_steps = config.steps_per_epoch * config.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[base_lr, base_lr * 10],
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        if config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}", total=self.config.steps_per_epoch
        )

        for step, (images, inp_ids, labels, lengths) in enumerate(pbar):
            if step >= self.config.steps_per_epoch:
                break

            # non_blocking=True allows async transfer while CPU prepares the next batch
            images = images.to(self.device, non_blocking=True)
            inp_ids = inp_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            B = inp_ids.shape[0]
            sos_token = torch.full(
                (B, 1), self.eos_token_id, dtype=torch.long, device=self.device
            )
            model_inp = torch.cat([sos_token, inp_ids[:, :-1]], dim=1)

            self.optimizer.zero_grad(
                set_to_none=True
            )  # Slightly faster than zero_grad()

            # --- HPC: Mixed Precision Forward Pass ---
            # Routes matrix multiplications directly to physical Tensor Cores
            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                logits = self.model(images, model_inp, lengths)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )

            # --- HPC: Scaled Backward Pass ---
            self.scaler.scale(loss).backward()

            # Unscale before clipping gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            lrs = self.scheduler.get_last_lr()
            pbar.set_postfix({"loss": loss.item(), "lr_gpt": lrs[0], "lr_vis": lrs[1]})

            # Throttle W&B logging to prevent I/O blocking
            if self.config.use_wandb and step % 50 == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "lr_gpt": lrs[0],
                        "lr_vis": lrs[1],
                        "global_step": epoch * self.config.steps_per_epoch + step,
                    }
                )

            if step > 0 and step % 1000 == 0:
                self.log_predictions(f"{epoch}_step_{step}")
                self.model.train()

        self.log_predictions(epoch)
        return total_loss / self.config.steps_per_epoch

    # [REST OF TRAINER REMAINS IDENTICAL: unnormalize_image, generate_controlled, evaluate, save, load, draw_text]
    def unnormalize_image(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)

    def generate_controlled(
        self, images, target_lengths, temperature=1.0, top_k=50, repetition_penalty=1.2
    ):
        self.model.eval()
        B = images.shape[0]
        device = self.device
        with torch.no_grad():
            vision_dict = self.model.vision_encoder.forward_features(images)
            raw_visual = vision_dict["x_norm_patchtokens"]
            visual_embeds = self.model.visual_mapper(raw_visual)
        input_ids = torch.full(
            (B, 1), self.eos_token_id, dtype=torch.long, device=device
        )
        current_lengths = torch.zeros(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values = None

        while not finished.all():
            with torch.no_grad():
                if past_key_values is None:
                    new_token_ids = input_ids
                    positions = torch.zeros((B, 1), dtype=torch.long, device=device)
                else:
                    new_token_ids = input_ids[:, -1:]
                    positions = current_lengths.unsqueeze(1)
                word_embeds = self.model.gpt2.base_model.model.transformer.wte(
                    new_token_ids
                )
                count_embeds = self.model.countdown_emb(positions, target_lengths)
                text_embeds = word_embeds + count_embeds
                if past_key_values is None:
                    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                else:
                    inputs_embeds = text_embeds
                outputs = self.model.gpt2(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                if repetition_penalty != 1.0:
                    for i in range(B):
                        unique_tokens = torch.unique(input_ids[i])
                        mask = unique_tokens != self.eos_token_id
                        unique_tokens = unique_tokens[mask]
                        if len(unique_tokens) > 0:
                            selected_logits = next_token_logits[i, unique_tokens]
                            next_token_logits[i, unique_tokens] = torch.where(
                                selected_logits < 0,
                                selected_logits * repetition_penalty,
                                selected_logits / repetition_penalty,
                            )
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_k_probs, _ = torch.topk(next_token_logits, top_k)
                    min_val = top_k_probs[:, -1].unsqueeze(-1)
                    next_token_logits[next_token_logits < min_val] = float("-inf")
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.eos_token_id), next_token
            )
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            current_lengths += (~finished).long()
            finished = current_lengths >= target_lengths
        return input_ids[:, 1:]

    def log_predictions(self, epoch, num_samples=4):
        if not self.config.use_wandb:
            return
        loader = self.val_loader if self.val_loader is not None else self.train_loader
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
            try:
                first_eos_index = full_gen.index(self.eos_token_id)
                actual_length = first_eos_index + 1
            except ValueError:
                actual_length = -1
            display_slice = (
                full_gen[:actual_length] if actual_length != -1 else full_gen
            )
            text = self.tokenizer.decode(display_slice)
            clean_text = text.replace("<|endoftext|>", "[EOS]")
            tokens_readable = [
                f"'{self.tokenizer.decode([t])}'" if t != self.eos_token_id else "[EOS]"
                for t in display_slice
            ]
            tokens_gt_readable = [
                (
                    f"'{self.tokenizer.decode([gt])}'"
                    if gt != self.eos_token_id
                    else "[EOS]"
                )
                for gt in inp_ids[i].tolist()[:target_k]
            ]
            if actual_length == target_k:
                status = "SUCCESS: Exact Length Match"
            elif actual_length == -1:
                status = "FAIL: No EOS generated"
            else:
                status = f"FAIL: Stopped at {actual_length} (Goal: {target_k})"
            gt_seq = inp_ids[i].tolist()[:target_k]
            gt_text = self.tokenizer.decode(gt_seq).replace("<|endoftext|>", "[EOS]")
            caption = f"Goal: {target_k} tokens\nStatus: {status}\n\nPrediction ({len(tokens_readable)} toks):\n{clean_text}\nBreakdown: {tokens_readable}\n\nGround Truth:\n{gt_text}\nGround Truth Breakdown: {tokens_gt_readable}\n"
            wandb_images.append(wandb.Image(vis_images[i], caption=caption))
        wandb.log({"Validation Samples": wandb_images, "epoch": epoch})

    def evaluate(self, evaluator, num_batches=None):
        report = evaluator.compute_metrics(num_batches=num_batches)
        if self.config.use_wandb:
            wandb.log({f"Eval/{k}": v for k, v in report.items()})
        print("Evaluation Report:", report)
        return report

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
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"] + 1
