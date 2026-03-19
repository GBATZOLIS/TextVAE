import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.distributed as dist


class DecoderTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        self.is_distributed = dist.is_initialized()
        self.is_main_process = (not self.is_distributed) or (dist.get_rank() == 0)

        self.use_amp = getattr(config, "use_amp", False)
        self.amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # We only optimize the DiT Transformer parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)

        total_steps = config.steps_per_epoch * config.epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

        if self.is_main_process and config.use_wandb:
            wandb.init(project="planning-autoencoder-decoder", config=vars(config))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        iterator = (
            tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                total=self.config.steps_per_epoch,
            )
            if self.is_main_process
            else self.train_loader
        )

        for step, (images, captions, input_ids, attention_mask) in enumerate(iterator):
            if step >= self.config.steps_per_epoch:
                break

            images = images.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss = self.model(images, input_ids, attention_mask)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if self.is_main_process:
                iterator.set_postfix({"loss": loss.item()})
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "global_step": epoch * self.config.steps_per_epoch + step,
                    }
                )

            # --- DDP FIX: Force all GPUs to wait before logging generations ---
            if step > 0 and step % 1000 == 0:
                if self.is_distributed:
                    dist.barrier()
                self.log_predictions(f"{epoch}_step_{step}")
                self.model.train()

        # End of epoch logging
        if self.is_distributed:
            dist.barrier()
        self.log_predictions(epoch)

        return total_loss / self.config.steps_per_epoch

    def log_predictions(self, epoch, num_samples=2):
        if not self.is_main_process or not self.config.use_wandb:
            return

        self.model.eval()
        try:
            images, captions, _, _ = next(iter(self.val_loader))
        except StopIteration:
            return

        texts_to_generate = captions[:num_samples]
        gt_images = images[:num_samples]

        # Convert GT [-1, 1] images to [0, 1] for logging
        gt_images = (gt_images + 1.0) / 2.0

        # Generate images from text
        generated_images = (
            self.model.module.generate(texts_to_generate)
            if hasattr(self.model, "module")
            else self.model.generate(texts_to_generate)
        )

        wandb_images = []
        for i in range(len(texts_to_generate)):
            prompt = texts_to_generate[i]
            caption = f"Prompt: {prompt[:200]}..."

            wandb_images.append(wandb.Image(gt_images[i].cpu(), caption="Ground Truth"))
            wandb_images.append(wandb.Image(generated_images[i], caption=caption))

        wandb.log({"Reconstructions": wandb_images, "epoch": epoch})

    def save(self, path, epoch):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
