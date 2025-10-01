import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
import os
import wandb
from torchtext.data.metrics import bleu_score

from shapes3d import Shapes3DDataset

# Assuming your model is in a file named 'language_model.py'
from language_model import ImageToTextModel


def save_checkpoint(model, optimizer, epoch, bleu, path):
    """Saves model and optimizer state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "bleu": bleu,
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def train_epoch(
    model, dataloader, optimizer, criterion, device, grad_clip, global_step
):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # UPDATED: Use the correct keys from the collate function
        images, targets = batch["image_lang"].to(device), batch["tokens"].to(device)

        outputs = model(images, targets)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"train/grad_norm": grad_norm.item(), "step": global_step})
        global_step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(dataloader), global_step


@torch.no_grad()
def evaluate(model, dataloader, criterion, dataset, device, epoch):
    model.eval()
    total_loss = 0.0
    all_candidates, all_references = [], []
    table = wandb.Table(columns=["Epoch", "Image", "Ground Truth", "Prediction"])

    pbar = tqdm(dataloader, desc="Evaluating")
    for i, batch in enumerate(pbar):
        # UPDATED: Use the correct keys from the collate function
        images, targets = batch["image_lang"].to(device), batch["tokens"].to(device)

        loss = criterion(
            model(images, targets).view(-1, model.decoder.linear.out_features),
            targets.view(-1),
        )
        total_loss += loss.item()

        predicted_strings = model.predict(images, dataset)

        for j in range(len(predicted_strings)):
            gt_str = dataset.detokenize(targets[j])

            all_candidates.append(predicted_strings[j].split(" | "))
            all_references.append([gt_str.split(" | ")])

            if i == 0 and j < 16:
                img_to_log = images[j].cpu() * 0.5 + 0.5
                table.add_data(
                    epoch, wandb.Image(img_to_log), gt_str, predicted_strings[j]
                )

    avg_loss = total_loss / len(dataloader)
    bleu = bleu_score(all_candidates, all_references)
    return avg_loss, bleu, table


def main(args):
    wandb.init(project=args.project_name, config=args)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # UPDATED: Instantiate the unified dataset
    dataset = Shapes3DDataset(image_size_lang=config.image_size, split="train")
    train_size = int(0.9 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    pad_idx = dataset.token_to_idx["[PAD]"]

    # UPDATED: Collate function uses the correct keys from the dataset's __getitem__
    def collate_fn(data):
        images = torch.stack([item["image_lang"] for item in data])
        targets = [item["tokens"] for item in data]
        padded_targets = nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=pad_idx
        )
        return {"image_lang": images, "tokens": padded_targets}

    train_loader = DataLoader(
        train_set, config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, config.batch_size, num_workers=4, collate_fn=collate_fn
    )

    model = ImageToTextModel(
        config.embed_size, config.hidden_size, dataset.vocab_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs)

    wandb.watch(model, criterion, log="all", log_freq=100)
    best_bleu = 0.0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.grad_clip,
            global_step,
        )
        val_loss, val_bleu, results_table = evaluate(
            model, val_loader, criterion, dataset, device, epoch
        )
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val BLEU={val_bleu:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/bleu": val_bleu,
                "learning_rate": scheduler.get_last_lr()[0],
                "evaluation_samples": results_table,
            }
        )

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                f"{config.checkpoint_dir}/best_model.pth",
            )
        save_checkpoint(
            model, optimizer, epoch, val_bleu, f"{config.checkpoint_dir}/last_model.pth"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ImageToText model with Unified Dataset."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # ADDED: Argument to specify image size for the language model
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size for the language model."
    )
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--project_name", type=str, default="ImageToText-Unified-Pipeline"
    )
    args = parser.parse_args()
    main(args)
