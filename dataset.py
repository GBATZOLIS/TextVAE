import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
import random
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
        except (FileNotFoundError, OSError) as e:
            raise NotImplementedError(
                f"Failed to load image {img_path}. Prefer to not generate deceiving dummy data. Original error: {e}"
            )

        # 2. Text Processing
        full_tokens = self.tokenizer.encode(item["caption"])

        # --- KEY LOGIC: RANDOM TRUNCATION ---
        # We simulate different "Budget Constraints" for the same image.
        # Sometimes we want the full caption, sometimes just 2 words.

        if len(full_tokens) > 1:
            # Pick a random budget K between 1 and Full Length
            budget = random.randint(1, len(full_tokens))
            tokens = full_tokens[:budget]
        else:
            tokens = full_tokens

        # Add EOS exactly at the end of the budget
        tokens.append(self.eos_token)

        # Token Tensor: [w1, w2, ... w_k, EOS]
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # The Length includes the EOS token.
        # If tokens are [A, Cat, EOS], length is 3.
        # EOS is at index 2 (0-indexed).
        length = len(tokens)

        return image, token_tensor, length


def collate_fn(batch):
    images, tokens_list, lengths = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor(lengths)

    # 1. Input IDs: Pad with EOS (standard for GPT inputs)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=50256
    )

    # 2. Target Labels: Pad with -100 (Ignore Index)
    # This ensures the model is NOT penalized for what happens after the first EOS.
    # It prevents the "Multiple EOS" bug.
    labels = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=-100
    )

    return images, input_ids, labels, lengths
