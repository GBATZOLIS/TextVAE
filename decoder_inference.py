import os
import gc
import torch
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms  # Moved to top level

# Import your local configuration and modules
from decoder.decoder_config import DecoderConfig
from decoder.pretrained_decoder import SemanticDecoder
from decoder_dataset import StreamingDecoderDataset, collate_fn
from decoder.decoder_eval import DecoderEvaluator


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint_cleanly(model, checkpoint_path, device):
    """Loads a DDP-trained model checkpoint into a standard model cleanly."""

    # MEMORY FIX: Load to CPU first!
    # Checkpoints contain massive optimizer states. Loading them directly
    # to the GPU will immediately cause a 24GB RTX 4090 to OOM.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Strip 'module.' prefix created by DistributedDataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    # Load the weights (PyTorch handles moving them to the model's device)
    model.load_state_dict(cleaned_state_dict, strict=True)
    epoch = checkpoint.get("epoch", "Unknown")
    print(f"Successfully loaded checkpoint from Epoch {epoch}")

    # Forcefully clear the massive checkpoint dictionary out of System RAM
    del checkpoint
    del state_dict
    del cleaned_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return model


def save_visualizations(model, loader, output_dir="decoder_results", num_images=5):
    """Generates side-by-side ground truth vs generated comparisons with prompts."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # We will accumulate images by looping over the dataloader since batch_size=1
    real_images_list, captions_list = [], []
    iterator = iter(loader)

    for _ in range(num_images):
        try:
            real_images, captions, _, _ = next(iterator)
            real_images_list.append(real_images[0])
            captions_list.append(captions[0])
        except StopIteration:
            break

    # Stack back into a single tensor for generation
    real_images_batch = torch.stack(real_images_list)

    print(f"\nGenerating {len(captions_list)} visual samples for inspection...")
    with torch.no_grad():
        generated_pil_images = model.generate(captions_list)

    # Unnormalize real images from [-1, 1] to [0, 1] for PIL conversion
    real_images_norm = (real_images_batch + 1.0) / 2.0
    to_pil = transforms.ToPILImage()

    for i in range(len(captions_list)):
        prompt = captions_list[i]
        gt_pil = to_pil(real_images_norm[i].cpu())
        gen_pil = generated_pil_images[i]

        # Build side-by-side canvas
        w, h = gt_pil.size
        margin = 80
        canvas = Image.new("RGB", (w * 2, h + margin), (255, 255, 255))

        # Paste images
        canvas.paste(gt_pil, (0, margin))
        canvas.paste(gen_pil, (w, margin))

        # Draw Text
        draw = ImageDraw.Draw(canvas)
        try:
            # Try to load a standard readable font, fallback to default
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        header_text = "Left: Ground Truth   |   Right: Generated"
        draw.text((w - 150, 10), header_text, fill=(0, 0, 0), font=font)

        # Truncate prompt if it's absurdly long for the image header
        display_prompt = prompt if len(prompt) < 120 else prompt[:117] + "..."
        draw.text((10, 40), f"Prompt: {display_prompt}", fill=(50, 50, 50), font=font)

        save_path = os.path.join(output_dir, f"sample_{i}.jpg")
        canvas.save(save_path)

    print(f"Saved visual comparisons to ./{output_dir}/")


def main():
    set_seed()

    # 1. Load config
    config = DecoderConfig()
    config.use_wandb = False

    # --- MEMORY FIX 1: Drop batch size to 1 ---
    # With T5-XXL + DiT + CLIP + Inception all in VRAM,
    # generation activations for larger batches will OOM a 24GB card.
    config.batch_size = 1

    # --- MEMORY FIX 2: Enable memory efficient formats ---
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(config.device)

    # 2. Setup Validation Dataset
    print("Loading dataset...")
    full_ds = StreamingDecoderDataset(config)

    # Create a strict, deterministic validation subset (first 500 images)
    val_indices = list(range(500))
    val_subset = Subset(full_ds, val_indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    # 3. Create model
    print("Instantiating Semantic Decoder...")
    model = SemanticDecoder(config).to(device)

    # 4. Load Checkpoint
    checkpoint_path = "/home/rg625/mnt/TextVAE/decoder_checkpoints/decoder_epoch_26.pt"  # Replace with your target
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    model = load_checkpoint_cleanly(model, checkpoint_path, device)

    # 5. Visual Inference
    save_visualizations(model, val_loader, output_dir="decoder_results", num_images=5)

    # --- MEMORY FIX 3: Aggressive Cache Clearing ---
    # Clear out generation activations before loading CLIP & Inception
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Quantitative Evaluation (CLIP + FID)
    evaluator = DecoderEvaluator(model, val_loader, device)

    # Run the sweep
    evaluator.compute_metrics(num_batches=None)


if __name__ == "__main__":
    main()
