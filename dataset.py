import torch
import itertools
import unittest.mock
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from datasets import load_dataset
from transformers import GPT2Tokenizer
import requests
from io import BytesIO
import torch.distributed as dist
from PIL import Image, ImageFile

# FIX: Prevent libjpeg/libpng from sending SIGABRT to the worker
# when it encounters a corrupted or truncated image from the web.
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --- HPC SAFETY: Prevent Decompression Bomb crashes on large web images ---
Image.MAX_IMAGE_PIXELS = None


class StreamingDenseCaptionDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = load_dataset(
            config.hf_dataset_path, split="train", cache_dir=config.hf_cache_dir
        )

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
            if "image" in item and isinstance(item["image"], Image.Image):
                image = self._robust_rgb_convert(item["image"])
            elif "image_url" in item and isinstance(item["image_url"], str):
                response = requests.get(item["image_url"], timeout=5)
                raw_image = Image.open(BytesIO(response.content))
                image = self._robust_rgb_convert(raw_image)
            elif "url" in item and isinstance(item["url"], str):
                response = requests.get(item["url"], timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                val = list(item.values())[0]
                if isinstance(val, Image.Image):
                    image = val.convert("RGB")
                else:
                    return None

            image_tensor = self.transform(image)

            caption = item.get("text", item.get("caption", ""))
            if isinstance(caption, list):
                caption = caption[0]

            caption = str(caption).strip()
            if not caption:
                return None

            encoded = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.config.max_len,
                return_tensors="pt",
            )
            ids = encoded.input_ids.squeeze(0)

            if ids[-1] != self.tokenizer.eos_token_id:
                if len(ids) == self.config.max_len:
                    ids[-1] = self.tokenizer.eos_token_id
                else:
                    ids = torch.cat([ids, torch.tensor([self.tokenizer.eos_token_id])])

            length = len(ids)
            return image_tensor, ids, length

        except Exception:
            return None

    def _robust_rgb_convert(self, image):
        """Safely converts any image format to RGB, replacing transparency with a white background."""
        # If it has an alpha channel or is a Palette image with transparency
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            # Convert to RGBA to safely extract the alpha channel
            image = image.convert("RGBA")
            # Create a solid white background
            background = Image.new("RGBA", image.size, (255, 255, 255))
            # Composite the image onto the white background, then drop the alpha channel
            return Image.alpha_composite(background, image).convert("RGB")
        else:
            # Normal images can just be converted directly
            return image.convert("RGB")

    def __iter__(self):
        worker_info = get_worker_info()

        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=None):
            base_iterator = iter(self.dataset)

            if worker_info is not None:
                # Multiply workers by world size to get total parallel streams
                total_workers = world_size * worker_info.num_workers
                # Offset the worker ID by the GPU's global rank
                global_worker_id = (
                    global_rank * worker_info.num_workers
                ) + worker_info.id

                base_iterator = itertools.islice(
                    base_iterator, global_worker_id, None, total_workers
                )
            else:
                # Fallback if num_workers=0 but we are using DDP (e.g., debugging)
                base_iterator = itertools.islice(
                    base_iterator, global_rank, None, world_size
                )

            for item in base_iterator:
                processed = self._process_item(item)
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
