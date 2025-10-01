import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import re
import logging
import random
from typing import Dict, List, Any, cast

# Note: Additional dependencies are required for this testing script.
# Please run: pip install datasets torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Dataset Class for Shapes3D ---
class Shapes3DDataset(Dataset):
    """
    Loads the Shapes3D dataset and provides tokenized, variable-order descriptions.
    Includes image augmentations for the training split.
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

    def __init__(self, image_size: int = 224, split: str = "train"):
        super().__init__()
        self.split = split
        self.image_size = image_size

        logger.info(f"Loading Shapes3D dataset for split: {self.split}...")
        self.hf_dataset = load_dataset("eurecom-ds/shapes3d", split=self.split)

        self._build_tokenizer_vocab()
        self._build_transforms()
        logger.info(f"Dataset setup complete for split: {self.split}")

    def _build_transforms(self):
        """Defines image transformations. Augmentations are only for the train set."""
        # For validation/testing, just resize and normalize
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _build_tokenizer_vocab(self):
        self.token_to_idx: Dict = {}
        self.idx_to_token: Dict = {}
        special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
        for token in special_tokens:
            self._add_token(token)

        for attr_name in self.ATTR_NAMES:
            if attr_name in self.ATTR_VALUES:
                for value_name in self.ATTR_VALUES[attr_name]:
                    self._add_token(f"{attr_name}={value_name}")
            else:
                for value_idx in range(self.NUM_BINS):
                    self._add_token(f"{attr_name}_bin={value_idx}")
        self.vocab_size = len(self.token_to_idx)

    def _add_token(self, token: str):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def detokenize(self, tokens: List[int] | torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        # Join tokens, skipping special tokens like SOS, EOS, PAD
        return " | ".join(
            self.idx_to_token.get(t_id, "[UNK]") for t_id in tokens if t_id > 2
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data_point = self.hf_dataset[index]
        img_pil = data_point["image"]
        img_tensor = self.transform(img_pil)

        # The detokenized string serves as our ground truth for the test
        ground_truth_str = self.detokenize(self._get_tokens(index))

        return {"image": img_tensor, "ground_truth": ground_truth_str}

    def _get_tokens(self, index: int) -> List[int]:
        """Helper to get tokenized attributes for an index."""
        data_point = self.hf_dataset[index]
        attributes = {}
        for attr_name in self.ATTR_NAMES:
            label_key = f"label_{attr_name}"
            value = data_point[label_key]
            if attr_name in self.ATTR_VALUES:
                attributes[attr_name] = self.ATTR_VALUES[attr_name][value]
            else:
                binned_value = min(int(value * self.NUM_BINS), self.NUM_BINS - 1)
                attributes[attr_name] = binned_value

        tokens = [self.token_to_idx["[SOS]"]]
        for name, value in attributes.items():
            token_str = (
                f"{name}={value}" if name in self.ATTR_VALUES else f"{name}_bin={value}"
            )
            tokens.append(self.token_to_idx.get(token_str, self.token_to_idx["[UNK]"]))
        tokens.append(self.token_to_idx["[EOS]"])
        return tokens


# --- VLM Functions ---
def load_model():
    """Loads the Llava model and processor from Hugging Face."""
    print(
        "Loading model... This may take a few minutes and requires a good internet connection."
    )
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully.")
    return model, processor


def predict_from_image(model, processor, raw_image: Image.Image) -> Dict | None:
    """
    Analyzes a PIL image using the Llava model and returns extracted attributes as a dictionary.
    """
    prompt = """You are an expert image analysis tool for the '3dshapes' dataset.
Analyze the provided image and extract its latent attributes.
The attributes are:
- floor_hue: A float value between 0.0 and 1.0.
- wall_hue: A float value between 0.0 and 1.0.
- object_hue: A float value between 0.0 and 1.0.
- scale: A float representing the object's size.
- shape: A string describing the object's shape (e.g., "sphere", "cube", "cylinder", "capsule").
- orientation: A float value representing the object's rotation in degrees.

Respond ONLY with a valid JSON object. Do not add any extra text, commentary, or markdown formatting."""
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Corrected the call to the processor by explicitly naming the 'text' argument.
    inputs = processor(text=full_prompt, images=raw_image, return_tensors="pt").to(
        "cuda"
    )
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response_text = processor.decode(output[0], skip_special_tokens=True)
    clean_response = response_text.split("ASSISTANT:")[-1].strip()

    json_match = re.search(r"\{.*\}", clean_response, re.DOTALL)
    if json_match:
        try:
            # Clean the JSON string by removing escaped underscores before parsing.
            json_string = json_match.group(0).replace("\\_", "_")
            return cast(Dict, json.loads(json_string))
        except json.JSONDecodeError:
            print(
                f"\n[Error] Model returned invalid JSON.\nRaw output: {clean_response}"
            )
            return None
    else:
        print(f"\n[Error] No JSON found in response.\nRaw output: {clean_response}")
        return None


def test_pipeline(model, processor, num_samples=5):
    """
    Tests the VLM pipeline by comparing its predictions against ground truth
    from the Shapes3D dataset.
    """
    print(f"\n--- Starting Pipeline Test on {num_samples} samples ---")
    # Corrected to use the available 'train' split
    dataset = Shapes3DDataset(split="train")

    unnormalize_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
            transforms.ToPILImage(),
        ]
    )

    for i in range(num_samples):
        index = random.randint(0, len(dataset) - 1)
        sample = dataset[index]

        pil_image = unnormalize_transform(sample["image"])
        ground_truth_str = sample["ground_truth"]

        print(f"\n--- Sample {i+1}/{num_samples} (Index: {index}) ---")
        print(f"Ground Truth: {ground_truth_str}")

        predicted_attrs = predict_from_image(model, processor, pil_image)

        if predicted_attrs:
            print("VLM Prediction:")
            print(json.dumps(predicted_attrs, indent=2))
        else:
            print("VLM Prediction: Failed to get a valid prediction.")
        print("-" * 35)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: This script requires a NVIDIA GPU and CUDA to run.")
        exit()

    llava_model, llava_processor = load_model()
    test_pipeline(llava_model, llava_processor, num_samples=5)
