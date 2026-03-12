import torch
import random
from encoder.encoder_config import EncoderConfig

# Make sure to import the correct architecture!
from encoder.pretrained_encoder import PlanningGPT2
from encoder.encoder_trainer import Trainer
from dataset import StreamingDenseCaptionDataset, collate_fn
from torch.utils.data import DataLoader
from encoder.encoder_eval import Evaluator


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# 1. Load config
config = EncoderConfig()
config.use_wandb = False  # Disable logging for inference
config.batch_size = 8  # Lower batch size for local inference testing

# Safety catch: Ensure steps_per_epoch exists so the Trainer's scheduler doesn't crash
if not hasattr(config, "steps_per_epoch"):
    config.steps_per_epoch = 5000

# 2. Setup Streaming Validation Data (The Fix)
print("Connecting to streaming dataset...")
val_ds = StreamingDenseCaptionDataset(config)

# For inference, we just take the first 500 items from the stream.
# No len(), no Subset(), no shuffle=True.
val_ds.dataset = val_ds.dataset.take(500)

val_loader = DataLoader(
    val_ds,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    num_workers=config.num_workers,
)

# 3. Create model
# IMPORTANT: This must match the architecture of the checkpoint you are loading.
model = PlanningGPT2(config).to(config.device)

# 4. Create Trainer
# Since this is pure inference, we don't need a train_loader.
# We just pass val_loader twice to satisfy the Trainer's __init__ requirements.
trainer = Trainer(model, train_loader=val_loader, val_loader=val_loader, config=config)

# 5. Load checkpoint
checkpoint_path = "/home/rg625/mnt/TextVAE/checkpoints/model_epoch_1.pt"
print(f"\nLoading checkpoint from {checkpoint_path}...")
trainer.load(checkpoint_path)

# 6. Run visual inference
print("\nGenerating visualizations...")
trainer.save_multi_length_visualizations(
    target_lengths=[5, 25, 50, 100, 150],
    output_dir="multi_length_results",
    num_images=5,
    temperature=0.3,  # 0.7 keeps the English grounded and prevents hallucinations
    top_k=50,
)

# 7. Run quantitative evaluation
evaluator = Evaluator(
    model=trainer.model,
    loader=trainer.val_loader,
    tokenizer=trainer.tokenizer,
    device=trainer.config.device,
)

# You can pass num_batches=10 if you want a quick test instead of the full 500
report = evaluator.compute_metrics(
    num_batches=None,
    plot_path="inference_score_vs_length.png",  # <--- Generates the plot here
)
print("\nFinal Report:")
print(report)
