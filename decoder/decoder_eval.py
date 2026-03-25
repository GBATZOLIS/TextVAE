import torch
from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms


class DecoderEvaluator:
    """
    Evaluates Text-to-Image models using two mandatory metrics:
    1. CLIP Score: Measures if the generated image matches the text prompt.
    2. FID Score: Measures if the generated images look realistic compared to the ground truth.
    """

    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

        print("Loading CLIP (ViT-B/32) for Semantic Alignment...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()

        print("Loading InceptionV3 for FID Evaluation...")
        # feature=2048 is standard for academic reporting of FID
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        self.to_tensor = transforms.ToTensor()

    def _calculate_clip_score(self, images, texts):
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
            # The diagonal of the logits matrix represents the score of the exact image-text pair
            scores = outputs.logits_per_image.diag().tolist()
        return scores

    def compute_metrics(self, num_batches=None):
        self.model.eval()
        clip_scores = []

        # Reset FID states for a fresh evaluation sweep
        self.fid.reset()

        # Handle potential DDP wrapping if evaluated immediately after training
        model_ptr = self.model.module if hasattr(self.model, "module") else self.model

        print("\nStarting Quantitative Sweep (CLIP + FID)...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.loader, desc="Evaluating Batches")):
                if num_batches and i >= num_batches:
                    break

                real_images, captions, _, _ = batch

                # 1. Generate Images (Diffusers Pipeline returns a list of PIL Images)
                generated_pil_images = model_ptr.generate(captions)

                # 2. Compute CLIP (Alignment)
                batch_scores = self._calculate_clip_score(
                    generated_pil_images, captions
                )
                clip_scores.extend(batch_scores)

                # 3. Format images for FID (Requires uint8 tensors [0, 255])
                # Real images are normalized to [-1, 1] for the VAE. Unnormalize to [0, 255]
                real_images_uint8 = (
                    ((real_images + 1.0) / 2.0 * 255).byte().to(self.device)
                )

                # Convert generated PIL images to [0, 255] tensors
                gen_tensors = torch.stack(
                    [self.to_tensor(img) for img in generated_pil_images]
                )
                gen_images_uint8 = (gen_tensors * 255).byte().to(self.device)

                # 4. Update FID distributions
                self.fid.update(real_images_uint8, real=True)
                self.fid.update(gen_images_uint8, real=False)

        # Compute final FID over the whole evaluated set
        print("Computing final Fréchet distance matrix (this may take a moment)...")
        fid_score = self.fid.compute().item()

        report = {
            "CLIP_Score": round(np.mean(clip_scores), 4),
            "FID_Score": round(fid_score, 4),
            "Samples_Evaluated": len(clip_scores),
        }

        print("\n" + "=" * 40)
        print("      DECODER EVALUATION REPORT")
        print("=" * 40)
        print(
            f" Semantic Alignment (CLIP) : {report['CLIP_Score']}  (Higher is better)"
        )
        print(f" Visual Fidelity (FID)     : {report['FID_Score']}  (Lower is better)")
        print("=" * 40)

        return report
