import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
import tiktoken  # Efficient BPE tokenizer


class ImageTextLengthDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=224, tokenizer_model="gpt2"):
        """
        Args:
            json_path: Path to JSON list of {image_id, caption, file_name}
            image_dir: Folder containing images
            image_size: Input size for the Vision Encoder
            tokenizer_model: Tiktoken encoding to use (default: gpt2)
        """
        self.image_dir = image_dir
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

        # We use tiktoken for efficient, standard tokenization
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        self.eos_token = 50256  # GPT-2 EOS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Load Image
        img_path = os.path.join(self.image_dir, item["file_name"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            print(f"Warning: Failed to load {img_path}, using black image.")
            image = torch.zeros((3, 224, 224))

        # 2. Process Text
        caption = item["caption"]
        tokens = self.tokenizer.encode(caption)

        # Append EOS
        tokens.append(self.eos_token)

        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # 3. Calculate Length (The Planning Signal)
        # The model needs to know the TOTAL length (including EOS) to count down
        length = len(tokens)

        return image, token_tensor, length


def collate_fn(batch):
    """
    Custom batch handling.
    Visuals: Stacked
    Text: Padded to max length in batch
    Lengths: Stacked
    """
    images, tokens_list, lengths = zip(*batch)

    images = torch.stack(images)
    lengths = torch.tensor(lengths)

    # Pad with EOS token (id 50256 for GPT2) or 0
    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=50256
    )

    return images, padded_tokens, lengths
