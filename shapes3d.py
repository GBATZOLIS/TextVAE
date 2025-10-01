import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
import random
from typing import Dict, List, Any, cast, Union


class Shapes3DDataset(Dataset):
    """
    Unified dataset for Shapes3D.
    Provides data in multiple formats for different models:
    - Token sequences for autoregressive models.
    - Flat condition vectors for diffusion models.
    """

    ATTR_NAMES = [
        "floor_hue",
        "wall_hue",
        "object_hue",
        "scale",
        "shape",
        "orientation",
    ]
    ATTR_VALUES = {"shape": ["Square", "Cylinder", "Sphere", "Dispenser"]}
    NUM_BINS = 10
    NUM_SHAPES = 4

    def __init__(self, image_size_lang=224, image_size_diff=224, split="train"):
        super().__init__()
        self.split = split
        self.hf_dataset = load_dataset("eurecom-ds/shapes3d", split=self.split)

        # Transformations for the language model (image-to-text)
        self.transform_lang = self._build_lang_transforms(image_size_lang)
        # Transformations for the diffusion model (text-to-image)
        self.transform_diff = self._build_diff_transforms(image_size_diff)

        self._build_tokenizer_vocab()

    def _build_lang_transforms(self, image_size):
        """Builds transforms for the autoregressive model, including augmentations."""
        if self.split == "train":
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                    ),
                    transforms.RandomAffine(
                        degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def _build_diff_transforms(self, image_size):
        """Builds transforms for the diffusion model, normalizing to [-1, 1]."""
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def _build_tokenizer_vocab(self):
        self.token_to_idx = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        for name in self.ATTR_NAMES:
            if name in self.ATTR_VALUES:
                for val in self.ATTR_VALUES[name]:
                    self._add_token(f"{name}={val}")
            else:
                for i in range(self.NUM_BINS):
                    self._add_token(f"{name}_bin={i}")
        self.vocab_size = len(self.token_to_idx)

    def _add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def get_attributes(self, index: int) -> Dict[str, Union[int, str]]:
        item = self.hf_dataset[index]
        attrs: Dict[str, Union[int, str]] = {}
        for name in self.ATTR_NAMES:
            val = item[f"label_{name}"]
            if name in self.ATTR_VALUES:
                # value is a string (like "red", "cube", etc.)
                attrs[name] = self.ATTR_VALUES[name][int(val)]
            else:
                # value is an int bin
                attrs[name] = min(int(val * self.NUM_BINS), self.NUM_BINS - 1)
        return attrs

    def tokenize(self, attributes: Dict[str, Any]) -> List[int]:
        tokens = [self.token_to_idx["[SOS]"]]
        items = list(attributes.items())
        if self.split == "train":
            random.shuffle(items)
        for name, value in items:
            token_str = (
                f"{name}={value}" if name in self.ATTR_VALUES else f"{name}_bin={value}"
            )
            tokens.append(self.token_to_idx.get(token_str, self.token_to_idx["[UNK]"]))
        tokens.append(self.token_to_idx["[EOS]"])
        return cast(List[int], tokens)  # type: ignore

    def detokenize(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        # Filter out special tokens before joining
        special_ids = {0, 1, 2, 3}
        return " | ".join(
            self.idx_to_token.get(t_id, "[UNK]")
            for t_id in tokens
            if t_id not in special_ids
        )

    def get_condition_vector(self, index: int) -> torch.Tensor:
        item = self.hf_dataset[index]
        fh = item["label_floor_hue"] * 2 - 1
        wh = item["label_wall_hue"] * 2 - 1
        oh = item["label_object_hue"] * 2 - 1
        sc = item["label_scale"] * 2 - 1
        ori = item["label_orientation"] * 2 - 1
        shape_one_hot = torch.nn.functional.one_hot(
            torch.tensor(int(item["label_shape"])), self.NUM_SHAPES
        )
        return torch.cat([torch.tensor([fh, wh, oh, sc, ori]), shape_one_hot.float()])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        raw_image = self.hf_dataset[index]["image"]

        # Get all representations
        img_lang = self.transform_lang(raw_image)
        img_diff = self.transform_diff(raw_image)
        tokens = torch.LongTensor(self.tokenize(self.get_attributes(index)))
        condition_vector = self.get_condition_vector(index)

        return {
            "image_lang": img_lang,
            "image_diff": img_diff,
            "tokens": tokens,
            "condition": condition_vector,
        }
