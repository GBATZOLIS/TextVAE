import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from datasets import load_dataset
from transformers import GPT2Tokenizer
from PIL import Image
import requests
from io import BytesIO
import itertools
import unittest.mock

# --- HPC SAFETY: Prevent Decompression Bomb crashes on large web images ---
Image.MAX_IMAGE_PIXELS = None

class StreamingDenseCaptionDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = load_dataset(config.hf_dataset_path, split="train", streaming=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_item(self, item):
        try:
            if 'image' in item and isinstance(item['image'], Image.Image):
                image = item['image'].convert("RGB")
            elif 'image_url' in item and isinstance(item['image_url'], str):
                response = requests.get(item['image_url'], timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            elif 'url' in item and isinstance(item['url'], str):
                response = requests.get(item['url'], timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                val = list(item.values())[0]
                if isinstance(val, Image.Image):
                    image = val.convert("RGB")
                else:
                    return None

            image_tensor = self.transform(image)

            caption = item.get('text', item.get('caption', ''))
            if isinstance(caption, list):
                caption = caption[0]
            
            caption = str(caption).strip()
            if not caption:
                return None

            encoded = self.tokenizer(
                caption, 
                truncation=True, 
                max_length=self.config.max_len, 
                return_tensors="pt"
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

    def __iter__(self):
        worker_info = get_worker_info()
        
        # 1. Blindfold Hugging Face so it doesn't crash trying to auto-shard a .skip() dataset
        # FIX: We MUST keep the entire iteration loop indented inside this 'with' block!
        with unittest.mock.patch('torch.utils.data.get_worker_info', return_value=None):
            base_iterator = iter(self.dataset)

            # 2. Apply our own robust native sharding manually
            if worker_info is not None:
                # islice will skip metadata items so each worker gets a unique, non-overlapping stream
                base_iterator = itertools.islice(base_iterator, worker_info.id, None, worker_info.num_workers)

            # 3. Process items normally (Downloading the images only happens for assigned items!)
            for item in base_iterator:
                processed = self._process_item(item)
                if processed is not None:
                    yield processed

def collate_fn(batch):
    images, tokens_list, lengths = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor(lengths)

    input_ids = torch.nn.utils.rnn.pad_sequence(tokens_list, batch_first=True, padding_value=50256)
    labels = torch.nn.utils.rnn.pad_sequence(tokens_list, batch_first=True, padding_value=-100)

    return images, input_ids, labels, lengths