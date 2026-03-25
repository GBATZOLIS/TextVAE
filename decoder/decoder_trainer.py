import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.distributed as dist
from PIL import Image
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


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
        self.amp_dtype = torch.bfloat16
        # ONLY enable scaler if using float16. BF16 does not need it.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16)
        )

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)

        # LEARNING RATE WARMUP
        warmup_steps = 1000
        total_steps = config.steps_per_epoch * config.epochs

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        if self.is_main_process and config.use_wandb:
            wandb.init(project="planning-autoencoder", config=vars(config))

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
                # loss = self.model(images, input_ids, attention_mask) # PixArt
                loss = self.model(images, captions)  # FLUX

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # 1. Capture the exact gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if self.is_main_process:
                iterator.set_postfix({"loss": loss.item()})

                # 2. Calculate the total parameter norm (only for trainable LoRA weights)
                with torch.no_grad():
                    param_norm = torch.sqrt(
                        sum(
                            torch.sum(p**2)
                            for p in self.model.parameters()
                            if p.requires_grad
                        )
                    )

                # 3. Log them to Weights & Biases!
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(),  # <--- ADDED
                        "param_norm": param_norm.item(),  # <--- ADDED
                        "global_step": epoch * self.config.steps_per_epoch + step,
                    }
                )

            if step > 0 and step % 2500 == 0:
                if self.is_distributed:
                    dist.barrier()
                self.log_predictions(f"{epoch}_step_{step}")
                self.model.train()

        if self.is_distributed:
            dist.barrier()
        self.log_predictions(epoch)

        return total_loss / self.config.steps_per_epoch

    def log_predictions(self, epoch, num_samples=4):
        if not self.is_main_process or not self.config.use_wandb:
            return

        self.model.eval()
        try:
            images, captions, _, _ = next(iter(self.val_loader))
        except StopIteration:
            return

        texts_to_generate = captions[:num_samples]
        gt_images = images[:num_samples]
        gt_images = (gt_images + 1.0) / 2.0

        model_ptr = self.model.module if hasattr(self.model, "module") else self.model
        generated_images = model_ptr.generate(texts_to_generate)

        wandb_images = []
        to_pil = transforms.ToPILImage()

        for i in range(len(texts_to_generate)):
            full_prompt = texts_to_generate[i]
            gt_pil = to_pil(gt_images[i].cpu())
            gen_pil = generated_images[i]

            w, h = gt_pil.size
            combined_img = Image.new("RGB", (w * 2, h))
            combined_img.paste(gt_pil, (0, 0))
            combined_img.paste(gen_pil, (w, 0))

            display_caption = f"Left: Real | Right: Generated\n\nPrompt: {full_prompt}"
            wandb_images.append(wandb.Image(combined_img, caption=display_caption))

        wandb.log({"Evaluation Samples": wandb_images, "epoch": epoch})

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
