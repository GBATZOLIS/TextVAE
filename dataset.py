import os
import json
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from transformers import PreTrainedTokenizerFast
import torch.distributed as dist
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ProgressiveDataset(IterableDataset):
    def __init__(self, config, tokenizer: PreTrainedTokenizerFast):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        with open(config.data_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _robust_rgb_convert(self, image):
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255))
            return Image.alpha_composite(background, image).convert("RGB")
        else:
            return image.convert("RGB")

    def _process_item(self, item):
        try:
            image_path = os.path.join(self.config.image_dir, item["local_filename"])
            image = self._robust_rgb_convert(Image.open(image_path))
            image_tensor = self.transform(image)

            # Use original high-quality caption for Phase B
            caption = item.get("original", str(item.get("medium", ""))).strip()
            if not caption:
                return None

            target_len = len(
                self.tokenizer(caption, add_special_tokens=False).input_ids
            )

            # PHASE B CURRICULUM: Simple unconditional description
            prompt = "<|im_start|>user\nDescribe this image using a highly detailed, flowing, and eloquent narrative style.<|im_end|>\n<|im_start|>assistant\n"
            full_text = prompt + caption + "<|im_end|>"

            encoded = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_encoded = self.tokenizer(prompt, return_tensors="pt")

            input_ids = encoded.input_ids.squeeze(0)
            attention_mask = encoded.attention_mask.squeeze(0)

            labels = input_ids.clone()
            labels[: prompt_encoded.input_ids.shape[1]] = -100
            labels[attention_mask == 0] = -100

            return {
                "images": image_tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "target_lengths": torch.tensor(target_len, dtype=torch.long),
            }
        except Exception:
            return None

    def __iter__(self):
        worker_info = get_worker_info()
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        total_streams = world_size * num_workers
        global_stream_id = (global_rank * num_workers) + worker_id

        for i in range(global_stream_id, len(self.dataset), total_streams):
            processed = self._process_item(self.dataset[i])
            if processed is not None:
                yield processed
