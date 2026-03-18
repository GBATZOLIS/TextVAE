import torch
import itertools
import unittest.mock
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from datasets import load_dataset
from transformers import T5Tokenizer
import requests
from io import BytesIO
import torch.distributed as dist
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class StreamingDecoderDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = load_dataset(
            config.hf_dataset_path, split="train", cache_dir=config.hf_cache_dir
        )

        # Diffusers VAE expects [-1, 1] normalization, NOT ImageNet stats!
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.img_size, config.img_size)),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            config.model_id, subfolder="tokenizer"
        )

    def _process_item(self, item):
        try:
            # Safely handle image extraction (same robust logic as encoder)
            if "image" in item and isinstance(item["image"], Image.Image):
                image = self._robust_rgb_convert(item["image"])
            elif "image_url" in item and isinstance(item["image_url"], str):
                response = requests.get(item["image_url"], timeout=5)
                image = self._robust_rgb_convert(Image.open(BytesIO(response.content)))
            else:
                return None

            image_tensor = self.transform(image)

            caption = item.get("text", item.get("caption", ""))
            if isinstance(caption, list):
                caption = caption[0]
            caption = str(caption).strip()

            if not caption:
                return None

            # T5 Tokenization
            encoded = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_len,
                return_tensors="pt",
            )

            # Extract raw string, input_ids, and attention mask
            return (
                image_tensor,
                caption,
                encoded.input_ids.squeeze(0),
                encoded.attention_mask.squeeze(0),
            )

        except Exception:
            return None

    def _robust_rgb_convert(self, image):
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

        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=None):
            base_iterator = iter(self.dataset)
            if worker_info is not None:
                total_workers = world_size * worker_info.num_workers
                global_worker_id = (
                    global_rank * worker_info.num_workers
                ) + worker_info.id
                base_iterator = itertools.islice(
                    base_iterator, global_worker_id, None, total_workers
                )
            else:
                base_iterator = itertools.islice(
                    base_iterator, global_rank, None, world_size
                )

            for item in base_iterator:
                processed = self._process_item(item)
                if processed is not None:
                    yield processed


def collate_fn(batch):
    images, captions, input_ids_list, attn_masks_list = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attn_masks_list)
    return images, list(captions), input_ids, attention_masks
