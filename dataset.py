import os
import json
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from transformers import GPT2Tokenizer
import torch.distributed as dist
from PIL import Image, ImageFile

# FIX: Prevent libjpeg/libpng from sending SIGABRT to the worker
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class StreamingDenseCaptionDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- HPC LOCAL PATHS ---
        self.data_path = "/home/rg625/datasets/pixmo_ready.jsonl"
        self.image_dir = "/home/rg625/datasets/pixmo_images"

        # Load the index completely into RAM (very fast for ~600k rows)
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

        self.transform = transforms.Compose(
            [
                transforms.Resize((config.img_size, config.img_size)),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_item(self, item):
        try:
            # 1. Blistering fast local NVMe read
            image_path = os.path.join(self.image_dir, item["local_filename"])
            raw_image = Image.open(image_path)
            image = self._robust_rgb_convert(raw_image)
            image_tensor = self.transform(image)

            # 2. LENGTH STRATIFICATION (The Brains of the Operation)
            # Randomly pick short, medium, or long
            choice = random.random()
            if choice < 0.33:
                caption = item.get("short", "")
            elif choice < 0.66:
                caption = item.get("medium", "")
            else:
                caption = item.get("long", "")

            # Fallback to the original if the augmentation script glitched on a row
            if not caption:
                caption = item.get("original", "")

            caption = str(caption).strip()
            if not caption:
                return None

            # 3. Tokenize and pad
            encoded = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.config.max_len,
                return_tensors="pt",
            )
            ids = encoded.input_ids.squeeze(0)

            # Ensure the EOS token is appended if it's not already there
            if ids[-1] != self.tokenizer.eos_token_id:
                if len(ids) == self.config.max_len:
                    ids[-1] = self.tokenizer.eos_token_id
                else:
                    ids = torch.cat([ids, torch.tensor([self.tokenizer.eos_token_id])])

            length = len(ids)
            return image_tensor, ids, length

        except Exception:
            # Silently drop corrupted images or missing files during training
            return None

    def _robust_rgb_convert(self, image):
        """Safely converts any image format to RGB."""
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255))
            return Image.alpha_composite(background, image).convert("RGB")
        else:
            return image.convert("RGB")

    def __iter__(self):
        worker_info = get_worker_info()
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Calculate total parallel streams across all GPUs and CPUs
        total_streams = world_size * num_workers
        global_stream_id = (global_rank * num_workers) + worker_id

        # Safely shard the local list across all workers using step slicing.
        # This guarantees no two GPUs or workers ever process the same image.
        for i in range(global_stream_id, len(self.dataset), total_streams):
            processed = self._process_item(self.dataset[i])
            if processed is not None:
                yield processed


def collate_fn(batch):
    images, tokens_list, lengths = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor(lengths)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=50256
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=-100
    )

    attention_mask = (input_ids != 50256).long()

    return images, input_ids, labels, lengths, attention_mask
