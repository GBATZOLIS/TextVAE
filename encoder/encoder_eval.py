import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# --- Import our decoupled visualizer ---
from .visualizations import plot_metrics_vs_length, plot_forced_length_sweep

try:
    from rouge_score import rouge_scorer

    HAS_ROUGE_SCORE = True
except ImportError:
    HAS_ROUGE_SCORE = False
    print("WARNING: 'rouge_score' not installed.")

try:
    from bert_score import score as bert_score_fn

    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False
    print("WARNING: 'bert_score' not installed.")

required_nltk_packages = ["punkt", "wordnet", "omw-1.4"]
for pkg in required_nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


class Evaluator:
    def __init__(self, model, loader, tokenizer, device, generate_fn):
        self.model = model
        self.loader = loader
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = 50256
        self.generate_batch = generate_fn

        if HAS_ROUGE_SCORE:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()

    def _calculate_batch_clip_score(self, images, generated_texts):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        unnorm_images = torch.clamp(images * std + mean, 0, 1)
        unnorm_images_uint8 = (
            unnorm_images.cpu().permute(0, 2, 3, 1).numpy() * 255
        ).astype(np.uint8)

        text_inputs = self.clip_processor.tokenizer(
            generated_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = self.clip_processor.image_processor(
            images=list(unnorm_images_uint8), return_tensors="pt"
        )

        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": image_inputs["pixel_values"].to(self.device),
        }

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            scores = outputs.logits_per_image.diag().tolist()
        return scores

    def calculate_rouge_l_fast(self, ref_text, hyp_text):
        if not HAS_ROUGE_SCORE or not ref_text or not hyp_text:
            return 0.0
        return self.rouge_scorer.score(ref_text, hyp_text)["rougeL"].fmeasure

    def compute_metrics(self, num_batches=None, plot_path="score_vs_length.png"):
        self.model.eval()
        all_references, all_hypotheses = [], []
        all_refs_texts, all_hyps_texts = [], []
        meteor_scores, rouge_scores, clip_scores = [], [], []
        target_lengths, actual_lengths = [], []
        exact_matches, total_samples = 0, 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.loader, desc="Eval Iteration")):
                if num_batches and i >= num_batches:
                    break

                images, inp_ids, target_lens = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[3].to(self.device),
                )
                gen_ids = self.generate_batch(
                    images, target_lens, temperature=1.0, top_k=50
                )
                batch_gen_texts = []

                for j in range(len(images)):
                    tgt_len = target_lens[j].item()

                    pred_tokens_ids = gen_ids[j].tolist()
                    pred_content_ids = (
                        pred_tokens_ids[: pred_tokens_ids.index(self.eos_token_id)]
                        if self.eos_token_id in pred_tokens_ids
                        else pred_tokens_ids
                    )
                    actual_len = (
                        len(pred_content_ids) + 1
                        if self.eos_token_id in pred_tokens_ids
                        else len(pred_tokens_ids)
                    )

                    pred_text = self.tokenizer.decode(pred_content_ids).strip()
                    batch_gen_texts.append(pred_text if pred_text else "empty")

                    ref_tokens_ids = inp_ids[j].tolist()
                    ref_content_ids = (
                        ref_tokens_ids[: ref_tokens_ids.index(self.eos_token_id)]
                        if self.eos_token_id in ref_tokens_ids
                        else ref_tokens_ids
                    )
                    ref_text = self.tokenizer.decode(ref_content_ids).strip()

                    target_lengths.append(tgt_len)
                    actual_lengths.append(actual_len)
                    if actual_len == tgt_len:
                        exact_matches += 1
                    total_samples += 1

                    all_hypotheses.append(pred_text.split())
                    all_references.append([ref_text.split()])
                    all_hyps_texts.append(pred_text)
                    all_refs_texts.append(ref_text)

                    meteor_scores.append(
                        meteor_score([ref_text.split()], pred_text.split())
                    )

                    rouge_scores.append(
                        self.calculate_rouge_l_fast(ref_text, pred_text)
                    )

                clip_scores.extend(
                    self._calculate_batch_clip_score(images, batch_gen_texts)
                )

        if total_samples == 0:
            return {"Count": 0}

        mae = np.mean(np.abs(np.array(actual_lengths) - np.array(target_lengths)))
        accuracy = exact_matches / total_samples
        b4 = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method4,
        )

        bert_f1 = 0.0
        if HAS_BERT_SCORE and len(all_hyps_texts) > 0:
            _, _, F1 = bert_score_fn(
                all_hyps_texts,
                all_refs_texts,
                lang="en",
                verbose=False,
                device=self.device,
            )
            bert_f1 = F1.mean().item() * 100

        # --- Delegate Plotting ---
        plot_metrics_vs_length(
            target_lengths,
            all_references,
            all_hypotheses,
            meteor_scores,
            rouge_scores,
            plot_path,
        )

        return {
            "Control_Accuracy": round(accuracy * 100, 2),
            "Control_MAE": round(mae, 4),
            "BLEU_4": round(b4 * 100, 2),
            "METEOR": round(np.mean(meteor_scores) * 100, 2),
            "ROUGE_L": round(np.mean(rouge_scores) * 100, 2),
            "CLIPScore": round(np.mean(clip_scores), 2),
            "BERTScore_F1": round(bert_f1, 2),
            "Count": total_samples,
        }

    def evaluate_multiple_lengths(
        self,
        forced_lengths=[5, 15, 30, 50, 100],
        num_batches=4,
        plot_path="forced_length_scores.png",
    ):
        self.model.eval()
        cached_batches = []
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                if num_batches and i >= num_batches:
                    break
                cached_batches.append((batch[0].cpu(), batch[1].cpu()))

        if not cached_batches:
            return {}

        length_to_metrics = {}
        with torch.no_grad():
            for tgt_len in forced_lengths:
                all_refs, all_hyps, all_refs_texts, all_hyps_texts = [], [], [], []
                meteor_scores, rouge_scores, clip_scores = [], [], []

                for images, inp_ids in cached_batches:
                    images = images.to(self.device)
                    forced_lens_tensor = torch.full(
                        (images.shape[0],),
                        tgt_len,
                        dtype=torch.long,
                        device=self.device,
                    )
                    gen_ids = self.generate_batch(
                        images, forced_lens_tensor, temperature=1.0, top_k=50
                    )

                    batch_gen_texts = []
                    for j in range(images.shape[0]):
                        pred_tokens_ids = gen_ids[j].tolist()
                        pred_content_ids = (
                            pred_tokens_ids[: pred_tokens_ids.index(self.eos_token_id)]
                            if self.eos_token_id in pred_tokens_ids
                            else pred_tokens_ids
                        )
                        pred_text = self.tokenizer.decode(pred_content_ids).strip()
                        batch_gen_texts.append(pred_text if pred_text else "empty")

                        ref_tokens_ids = inp_ids[j].tolist()
                        ref_content_ids = (
                            ref_tokens_ids[: ref_tokens_ids.index(self.eos_token_id)]
                            if self.eos_token_id in ref_tokens_ids
                            else ref_tokens_ids
                        )
                        ref_text = self.tokenizer.decode(ref_content_ids).strip()

                        all_hyps.append(pred_text.split())
                        all_refs.append([ref_text.split()])
                        all_hyps_texts.append(pred_text)
                        all_refs_texts.append(ref_text)

                        meteor_scores.append(
                            meteor_score([ref_text.split()], pred_text.split())
                        )
                        rouge_scores.append(
                            self.calculate_rouge_l_fast(ref_text, pred_text)
                        )

                    clip_scores.extend(
                        self._calculate_batch_clip_score(images, batch_gen_texts)
                    )

                b4 = corpus_bleu(
                    all_refs,
                    all_hyps,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=SmoothingFunction().method4,
                )

                bert_f1 = 0.0
                if HAS_BERT_SCORE:
                    _, _, F1 = bert_score_fn(
                        all_hyps_texts,
                        all_refs_texts,
                        lang="en",
                        verbose=False,
                        device=self.device,
                    )
                    bert_f1 = F1.mean().item() * 100

                length_to_metrics[tgt_len] = {
                    "BLEU_4": b4 * 100,
                    "METEOR": np.mean(meteor_scores) * 100,
                    "ROUGE_L": np.mean(rouge_scores) * 100,
                    "CLIPScore": np.mean(clip_scores),
                    "BERTScore_F1": bert_f1,
                }
                print(
                    f"Forced Length {tgt_len:>3} | CLIPScore: {np.mean(clip_scores):>5.2f} | BERTScore: {bert_f1:>5.2f} | BLEU-4: {b4*100:>5.2f}"
                )

        # --- Delegate Plotting ---
        plot_forced_length_sweep(
            forced_lengths, length_to_metrics, HAS_BERT_SCORE, plot_path
        )

        return length_to_metrics

    def evaluate_sota_baseline(self, num_batches=None):
        self.model.eval()
        all_refs = []
        all_hyps = []
        all_refs_texts = []
        all_hyps_texts = []
        meteor_scores = []
        rouge_scores = []
        clip_scores = []
        total_samples = 0

        print("\nRunning SOTA Baseline Evaluation (Sampling from N(217, 74))...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.loader)):
                if num_batches and i >= num_batches:
                    break

                images = batch[0].to(self.device)
                inp_ids = batch[1].to(self.device)
                B = images.shape[0]

                sampled_lens = np.random.normal(loc=217.59, scale=74.16, size=B)
                sampled_lens = np.clip(sampled_lens, a_min=10, a_max=350)
                forced_lens_tensor = torch.tensor(
                    sampled_lens, dtype=torch.long, device=self.device
                )

                gen_ids = self.generate_batch(
                    images, forced_lens_tensor, temperature=1.0, top_k=50
                )
                batch_gen_texts = []

                for j in range(B):
                    pred_tokens_ids = gen_ids[j].tolist()
                    try:
                        eos_idx = pred_tokens_ids.index(self.eos_token_id)
                        pred_content_ids = pred_tokens_ids[:eos_idx]
                    except ValueError:
                        pred_content_ids = pred_tokens_ids
                    pred_text = self.tokenizer.decode(pred_content_ids).strip()
                    pred_words = pred_text.split()
                    batch_gen_texts.append(pred_text if pred_text else "empty")

                    ref_tokens_ids = inp_ids[j].tolist()
                    try:
                        ref_eos_idx = ref_tokens_ids.index(self.eos_token_id)
                        ref_content_ids = ref_tokens_ids[:ref_eos_idx]
                    except ValueError:
                        ref_content_ids = ref_tokens_ids
                    ref_text = self.tokenizer.decode(ref_content_ids).strip()
                    ref_words = ref_text.split()

                    all_hyps.append(pred_words)
                    all_refs.append([ref_words])
                    all_hyps_texts.append(pred_text)
                    all_refs_texts.append(ref_text)
                    total_samples += 1

                    meteor_scores.append(meteor_score([ref_words], pred_words))
                    rouge_scores.append(
                        self.calculate_rouge_l_fast(ref_text, pred_text)
                    )

                batch_clip_scores = self._calculate_batch_clip_score(
                    images, batch_gen_texts
                )
                clip_scores.extend(batch_clip_scores)

        smoothie = SmoothingFunction().method4
        b4 = corpus_bleu(
            all_refs,
            all_hyps,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )

        bert_f1 = 0.0
        if HAS_BERT_SCORE and len(all_hyps_texts) > 0:
            print("Calculating Semantic BERTScore...")
            P, R, F1 = bert_score_fn(
                all_hyps_texts,
                all_refs_texts,
                lang="en",
                verbose=False,
                device=self.device,
            )
            bert_f1 = F1.mean().item() * 100

        report = {
            "BLEU_4": round(b4 * 100, 2),
            "METEOR": round(np.mean(meteor_scores) * 100, 2),
            "ROUGE_L": round(np.mean(rouge_scores) * 100, 2),
            "CLIPScore": round(np.mean(clip_scores), 2),
            "BERTScore_F1": round(bert_f1, 2),
            "Count": total_samples,
        }

        print("\nSOTA Baseline Report (Distribution Matched):")
        print(report)
        return report

    def evaluate_multiple_lengths_and_plot(
        self,
        forced_lengths=[5, 15, 30, 50, 100],
        num_batches=4,
        plot_path="forced_length_scores.png",
    ):
        self.model.eval()
        print(f"\nFetching {num_batches} batches for Multi-Length Evaluation Sweep...")

        cached_batches = []
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                if num_batches and i >= num_batches:
                    break

                # Robust unpacking
                images, inp_ids = batch[0], batch[1]
                cached_batches.append((images.cpu(), inp_ids.cpu()))

        if not cached_batches:
            print("WARNING: No data found for multi-length evaluation.")
            return

        length_to_metrics = {}
        print(f"Evaluating model at forced lengths: {forced_lengths}\n")

        with torch.no_grad():
            for tgt_len in forced_lengths:
                all_refs = []
                all_hyps = []
                all_refs_texts = []
                all_hyps_texts = []
                meteor_scores = []
                rouge_scores = []
                clip_scores = []

                for images, inp_ids in cached_batches:
                    images = images.to(self.device)
                    B = images.shape[0]
                    forced_lens_tensor = torch.full(
                        (B,), tgt_len, dtype=torch.long, device=self.device
                    )

                    gen_ids = self.generate_batch(
                        images, forced_lens_tensor, temperature=1.0, top_k=50
                    )
                    batch_gen_texts = []

                    for j in range(B):
                        pred_tokens_ids = gen_ids[j].tolist()
                        try:
                            eos_idx = pred_tokens_ids.index(self.eos_token_id)
                            pred_content_ids = pred_tokens_ids[:eos_idx]
                        except ValueError:
                            pred_content_ids = pred_tokens_ids
                        pred_text = self.tokenizer.decode(pred_content_ids).strip()
                        pred_words = pred_text.split()
                        batch_gen_texts.append(pred_text if pred_text else "empty")

                        ref_tokens_ids = inp_ids[j].tolist()
                        try:
                            ref_eos_idx = ref_tokens_ids.index(self.eos_token_id)
                            ref_content_ids = ref_tokens_ids[:ref_eos_idx]
                        except ValueError:
                            ref_content_ids = ref_tokens_ids
                        ref_text = self.tokenizer.decode(ref_content_ids).strip()
                        ref_words = ref_text.split()

                        all_hyps.append(pred_words)
                        all_refs.append([ref_words])
                        all_hyps_texts.append(pred_text)
                        all_refs_texts.append(ref_text)

                        meteor_scores.append(meteor_score([ref_words], pred_words))
                        rouge_scores.append(
                            self.calculate_rouge_l_fast(ref_text, pred_text)
                        )

                    batch_clip_scores = self._calculate_batch_clip_score(
                        images, batch_gen_texts
                    )
                    clip_scores.extend(batch_clip_scores)

                smoothie = SmoothingFunction().method4
                b4 = corpus_bleu(
                    all_refs,
                    all_hyps,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothie,
                )

                bert_f1 = 0.0
                if HAS_BERT_SCORE:
                    P, R, F1 = bert_score_fn(
                        all_hyps_texts,
                        all_refs_texts,
                        lang="en",
                        verbose=False,
                        device=self.device,
                    )
                    bert_f1 = F1.mean().item() * 100

                length_to_metrics[tgt_len] = {
                    "BLEU_4": b4 * 100,
                    "METEOR": np.mean(meteor_scores) * 100,
                    "ROUGE_L": np.mean(rouge_scores) * 100,
                    "CLIPScore": np.mean(clip_scores),
                    "BERTScore_F1": bert_f1,
                }
                print(
                    f"Forced Length {tgt_len:>3} | CLIPScore: {np.mean(clip_scores):>5.2f} | BERTScore: {bert_f1:>5.2f} | BLEU-4: {b4*100:>5.2f}"
                )

        x_vals = forced_lengths
        _ = [length_to_metrics[length]["BLEU_4"] for length in x_vals]
        _ = [length_to_metrics[length]["CLIPScore"] for length in x_vals]
        _ = [length_to_metrics[length]["BERTScore_F1"] for length in x_vals]

        return length_to_metrics
