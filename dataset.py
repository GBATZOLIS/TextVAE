import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
import tiktoken


class ImageTextLengthDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=224, tokenizer_model="gpt2"):
        self.image_dir = image_dir

        # Load Data
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Generating dummy data.")
            self.data = [
                {
                    "file_name": "dummy.jpg",
                    "caption": "A photo of a cat sitting on a mat.",
                }
            ] * 100
        else:
            with open(json_path, "r") as f:
                self.data = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        self.eos_token = 50256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Image
        img_path = os.path.join(self.image_dir, item["file_name"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError):
            # Fallback for missing images to prevent crash
            image = torch.zeros((3, 224, 224))

        # 2. Text Processing
        # CHANGE: No random truncation. We take the full caption.
        tokens = self.tokenizer.encode(item["caption"])

        # Ensure we don't exceed a reasonable max length (e.g. 128) to prevent OOM
        # but try to keep the full sentence structure.
        max_capacity = 128
        if len(tokens) > max_capacity - 1:
            tokens = tokens[: max_capacity - 1]

        # Add EOS
        tokens.append(self.eos_token)

        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # The Target Length is the index where EOS is located.
        # If tokens = [A, Cat, EOS], length is 3.
        # Indices: 0, 1, 2.
        # We want the model to put EOS at position (length-1).
        length = len(tokens)

        return image, token_tensor, length


def collate_fn(batch):
    images, tokens_list, lengths = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor(lengths)

    # 1. Input IDs: Pad with EOS
    input_ids = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=50256
    )

    # 2. Target Labels
    labels = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=-100
    )

    return images, input_ids, labels, lengths
