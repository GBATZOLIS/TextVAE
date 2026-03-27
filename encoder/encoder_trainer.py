import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import tiktoken
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.distributed as dist

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token_id = 50256

        self.is_distributed = dist.is_initialized()
        self.is_main_process = (not self.is_distributed) or (dist.get_rank() == 0)
        self.world_size = dist.get_world_size() if self.is_distributed else 1

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
        
        if self.is_main_process and config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.is_main_process:
            iterator = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=self.config.steps_per_epoch)
        else:
            iterator = self.train_loader

        for step, (images, inp_ids, labels, lengths) in enumerate(iterator):
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

            self.optimizer.zero_grad(set_to_none=True)

            # --- HPC: Mixed Precision Forward Pass ---
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

            if self.is_main_process:
                iterator.set_postfix({"loss": loss.item()})
                
                # if step % 500 == 0 and self.config.use_wandb:
                lrs = self.scheduler.get_last_lr()
                wandb.log({
                    "train_loss": loss.item(),
                    "lr_gpt": lrs[0],
                    "lr_vis": lrs[1],
                    "global_step": epoch * self.config.steps_per_epoch + step,
                })

                if step > 0 and step % 1000 == 0:
                    self.log_predictions(f"{epoch}_step_{step}")
                    self.model.train() # Ensure it goes back to train mode!

        self.log_predictions(epoch)

        if self.is_distributed:
            loss_tensor = torch.tensor([total_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
        
        avg_loss = total_loss / (self.config.steps_per_epoch * dist.get_world_size())
        
        return avg_loss

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
        model_ptr = self.model.module if hasattr(self.model, "module") else self.model
    
        with torch.no_grad():
            # Use model_ptr instead of self.model
            outputs = model_ptr.vision_encoder(pixel_values=images)
            raw_visual = outputs.last_hidden_state[:, 1:, :]
            visual_embeds = model_ptr.visual_mapper(raw_visual)

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
                word_embeds = model_ptr.gpt2.base_model.model.transformer.wte(new_token_ids)
                count_embeds = model_ptr.countdown_emb(positions, target_lengths)
                
                text_embeds = word_embeds + count_embeds
                if past_key_values is None:
                    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                else:
                    inputs_embeds = text_embeds
                outputs = model_ptr.gpt2(
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
        if not self.is_main_process or not self.config.use_wandb:
            return
            
        loader = self.val_loader if self.val_loader is not None else self.train_loader
        try:
            images, inp_ids, _, lengths = next(iter(loader))
        except StopIteration:
            return
        images = images[:num_samples].to(self.device)
        lengths = lengths[:num_samples].to(self.device)

        # Hardcode temperature to 0.7 for automated W&B validation logging to prevent hallucinated sub-words
        generated_ids = self.generate_controlled(
            images, lengths, temperature=0.7, top_k=50
        )
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

    def generate_with_custom_length(
        self,
        target_length: int,
        num_samples: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        self.model.eval()
        if self.val_loader is None:
            raise ValueError("Validation loader not available.")
        images, _, _, _ = next(iter(self.val_loader))
        images = images[:num_samples].to(self.device)
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

    def evaluate(self, evaluator, num_batches=None):
        report = evaluator.compute_metrics(num_batches=num_batches)
        if self.config.use_wandb:
            wandb.log({f"Eval/{k}": v for k, v in report.items()})
        print("Evaluation Report:", report)
        return report

    def save(self, path, epoch):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
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

    def save_multi_length_visualizations(
        self,
        target_lengths=[5, 10, 15, 25, 50],
        output_dir="multi_length_results",
        num_images=8,
        temperature=1.0,
        top_k=50,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        if self.val_loader is None:
            raise ValueError("Validation loader not available.")

        images, inp_ids, _, gt_lengths = next(iter(self.val_loader))

        images = images[:num_images].to(self.device)
        inp_ids = inp_ids[:num_images]
        gt_lengths = gt_lengths[:num_images]

        vis_images = self.unnormalize_image(images).cpu()

        for i in range(images.size(0)):
            img_tensor = vis_images[i]
            img = transforms.ToPILImage()(img_tensor)
            caption_lines = []

            gt_len = gt_lengths[i].item()
            gt_tokens = inp_ids[i].tolist()[:gt_len]
            gt_text = self.tokenizer.decode(gt_tokens).replace("<|endoftext|>", "")
            caption_lines.append("Ground Truth:")
            caption_lines.append(gt_text)
            caption_lines.append("")

            for tgt in target_lengths:
                target_tensor = torch.tensor([tgt], device=self.device)
                generated_ids = self.generate_controlled(
                    images[i].unsqueeze(0),
                    target_tensor,
                    temperature=temperature,
                    top_k=top_k,
                )
                tokens = generated_ids[0].tolist()
                if self.eos_token_id in tokens:
                    eos_idx = tokens.index(self.eos_token_id)
                    tokens = tokens[: eos_idx + 1]

                text = self.tokenizer.decode(tokens).replace("<|endoftext|>", "")
                caption_lines.append(f"Target {tgt} (Actual {len(tokens)}):")
                caption_lines.append(text)
                caption_lines.append("")

            full_text = "\n".join(caption_lines)
            nice_img = self.draw_multiline_text_block(img, full_text, font_size=18)
            save_path = os.path.join(output_dir, f"sample_{i}.jpg")
            nice_img.save(save_path)

        print(f"\nSaved multi-length visualizations to {output_dir}")

    def draw_multiline_text_block(
        self, image, text, font_path=None, font_size=18, margin=20, line_spacing=6
    ):
        width, height = image.size

        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(image)
        wrapped_lines = []
        max_width = width - 2 * margin

        for line in text.split("\n"):
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]
                if line_width > max_width:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                wrapped_lines.append(current_line)
            if not words:
                wrapped_lines.append("")

        total_text_height = 0
        line_heights = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            total_text_height += line_height + line_spacing
            line_heights.append(line_height)

        total_text_height += margin

        new_img = Image.new("RGB", (width, height + total_text_height), (255, 255, 255))
        new_img.paste(image, (0, 0))
        draw_new = ImageDraw.Draw(new_img)

        y = height + margin
        for line, lh in zip(wrapped_lines, line_heights):
            draw_new.text((margin, y), line, fill=(0, 0, 0), font=font)
            y += lh + line_spacing

        return new_img
