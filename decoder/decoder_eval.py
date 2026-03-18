import torch
from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class DecoderEvaluator:
    """
    Evaluates the image generation capabilities by calculating the
    CLIP score between the ground-truth text and the generated image.
    """

    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

        print("Loading CLIP for Evaluation...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()

    def _calculate_clip_score(self, images, texts):
        # Clip processor expects PIL images or numpy arrays in uint8
        # The generated images from pipeline are PIL images
        inputs = self.clip_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Diagonal gives the exact image-text pair score
            scores = outputs.logits_per_image.diag().tolist()
        return scores

    def compute_metrics(self, num_batches=5):
        self.model.eval()
        clip_scores = []

        model_ptr = self.model.module if hasattr(self.model, "module") else self.model

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.loader, desc="Eval Iteration")):
                if num_batches and i >= num_batches:
                    break

                _, captions, _, _ = batch

                # Generate images from the ground truth captions
                generated_pil_images = model_ptr.generate(captions)

                # Calculate alignment score
                batch_scores = self._calculate_clip_score(
                    generated_pil_images, captions
                )
                clip_scores.extend(batch_scores)

        report = {
            "CLIPScore": round(np.mean(clip_scores), 2),
        }

        print("Decoder Evaluation Report:")
        print(report)
        return report
