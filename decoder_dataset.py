import os
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import T5Tokenizer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class StreamingDecoderDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.image_dir = config.image_dir
        self.split = split  # <--- NEW: Track if this is train or val

        with open(config.data_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

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

    def __len__(self):
        return len(self.dataset)

    def _robust_rgb_convert(self, image):
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255))
            return Image.alpha_composite(background, image).convert("RGB")
        return image.convert("RGB")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image_path = os.path.join(self.image_dir, item["local_filename"])
            raw_image = Image.open(image_path)
            image = self._robust_rgb_convert(raw_image)
            image_tensor = self.transform(image)

            # --- DETERMINISTIC FIX FOR WANDB ---
            if self.split == "train":
                # Randomly stratify lengths during training for robust text alignment
                choice = random.random()
                if choice < 0.33:
                    caption = item.get("short", "")
                elif choice < 0.66:
                    caption = item.get("medium", "")
                else:
                    caption = item.get("long", "")
            else:
                # Validation is locked to 'long' so WandB history sliders never break
                caption = item.get("long", "") or item.get("original", "")

            caption = str(caption).strip() or str(item.get("original", "")).strip()
            if not caption:
                caption = "a picture"

            encoded = self.tokenizer(
                caption,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_len,
                return_tensors="pt",
            )

            return (
                image_tensor,
                caption,
                encoded.input_ids.squeeze(0),
                encoded.attention_mask.squeeze(0),
            )

        except Exception:
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))


def collate_fn(batch):
    images, captions, input_ids_list, attn_masks_list = zip(*batch)
    images = torch.stack(images)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=0
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attn_masks_list, batch_first=True, padding_value=0
    )

    return images, list(captions), input_ids, attention_masks
