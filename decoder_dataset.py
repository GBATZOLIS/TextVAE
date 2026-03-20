import os
import json
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from transformers import T5Tokenizer
import torch.distributed as dist
from PIL import Image, ImageFile

# FIX: Prevent libjpeg/libpng from sending SIGABRT to the worker
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class StreamingDecoderDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- HPC LOCAL PATHS ---
        self.data_path = config.data_path
        self.image_dir = config.image_dir

        # Load the index completely into RAM
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

        # Diffusers VAE expects [-1, 1] normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.img_size, config.img_size)),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Load the T5 Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.model_id, subfolder="tokenizer"
        )

    def _process_item(self, item):
        try:
            # 1. Fast local NVMe image read
            image_path = os.path.join(self.image_dir, item["local_filename"])
            raw_image = Image.open(image_path)
            image = self._robust_rgb_convert(raw_image)
            image_tensor = self.transform(image)

            # 2. LENGTH STRATIFICATION
            choice = random.random()
            if choice < 0.33:
                caption = item.get("short", "")
            elif choice < 0.66:
                caption = item.get("medium", "")
            else:
                caption = item.get("long", "")

            # Fallback
            if not caption:
                caption = item.get("original", "")

            caption = str(caption).strip()
            if not caption:
                return None

            # 3. T5 Tokenization (Infinite Length - No Truncation or Padding)
            encoded = self.tokenizer(
                caption,
                truncation=False,  # The guillotine is completely removed
                return_tensors="pt",
            )

            return (
                image_tensor,
                caption,
                encoded.input_ids.squeeze(0),
                encoded.attention_mask.squeeze(0),
            )

        except Exception:
            return None

    def _robust_rgb_convert(self, image):
        """Safely converts any image format to RGB, dropping alpha channels if present."""
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255))
            return Image.alpha_composite(background, image).convert("RGB")
        return image.convert("RGB")

    def __iter__(self):
        worker_info = get_worker_info()
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Safely shard the local list across all workers and GPUs
        total_streams = world_size * num_workers
        global_stream_id = (global_rank * num_workers) + worker_id

        for i in range(global_stream_id, len(self.dataset), total_streams):
            processed = self._process_item(self.dataset[i])
            if processed is not None:
                yield processed


def collate_fn(batch):
    images, captions, input_ids_list, attn_masks_list = zip(*batch)
    images = torch.stack(images)

    # --- THE FIX: Dynamic Batch Padding ---
    # Pads the text tensors to match whatever the longest caption is in THIS specific batch
    # T5 uses 0 as its padding token ID
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attn_masks_list, batch_first=True, padding_value=0
    )

    return images, list(captions), input_ids, attention_masks
