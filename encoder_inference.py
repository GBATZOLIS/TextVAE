import torch
import os
import random
from encoder.encoder_config import EncoderConfig
from encoder.pretrained_encoder import PlanningGPT2
from encoder.encoder_trainer import Trainer
from dataset import ImageTextLengthDataset, collate_fn
from torch.utils.data import DataLoader, Subset

# 1. Load config
config = EncoderConfig()
config.use_wandb = False  # disable logging for inference


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

if not os.path.exists(config.json_path):
    raise FileNotFoundError(f"JSON file not found: {config.json_path}")

if not os.path.exists(config.img_dir):
    raise FileNotFoundError(f"Image directory not found: {config.img_dir}")

full_dataset = ImageTextLengthDataset(config.json_path, config.img_dir)

if len(full_dataset) == 0:
    raise ValueError("Dataset is empty. Aborting.")

print(f"Dataset size: {len(full_dataset)}")

# Split
indices = list(range(len(full_dataset)))
random.shuffle(indices)
split = int(0.9 * len(indices))

train_ds = Subset(full_dataset, indices[:split])
val_ds = Subset(full_dataset, indices[split:])

train_loader = DataLoader(
    train_ds,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)

val_loader = DataLoader(
    val_ds,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)

# 2. Create model
model = PlanningGPT2(config).to(config.device)

# 3. Create dummy trainer (no loaders needed)
trainer = Trainer(
    model, train_loader=train_loader, val_loader=val_loader, config=config
)

# 4. Load checkpoint
checkpoint_path = "/home/rg625/mnt/TextVAE/checkpoints/pretrained_epoch_250.pt"
trainer.load(checkpoint_path)

# 5. Run inference
trainer.generate_with_custom_length(target_length=10, num_samples=4)
