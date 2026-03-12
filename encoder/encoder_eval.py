import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict

required_nltk_packages = ["punkt", "wordnet", "omw-1.4"]
for pkg in required_nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)


class Evaluator:
    def __init__(self, model, loader, tokenizer, device):
        self.model = model
        self.loader = loader
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = 50256

    def compute_metrics(self, num_batches=None, plot_path="score_vs_length.png"):
        self.model.eval()
        all_references = []
        all_hypotheses = []
        meteor_scores = []
        rouge_scores = []
        target_lengths = []
        actual_lengths = []
        exact_matches = 0
        total_samples = 0

        print("Running Comprehensive Evaluation...")
        with torch.no_grad():
            for i, (images, inp_ids, _, lengths) in enumerate(tqdm(self.loader)):
                if num_batches and i >= num_batches:
                    break

                images = images.to(self.device)
                target_lens = lengths.to(self.device)
                gen_ids = self.generate_batch(images, target_lens)

                for j in range(len(images)):
                    tgt_len = target_lens[j].item()
                    pred_tokens_ids = gen_ids[j].tolist()
                    try:
                        eos_idx = pred_tokens_ids.index(self.eos_token_id)
                        actual_len = eos_idx + 1
                        pred_content_ids = pred_tokens_ids[:eos_idx]
                    except ValueError:
                        actual_len = len(pred_tokens_ids)
                        pred_content_ids = pred_tokens_ids

                    pred_text = self.tokenizer.decode(pred_content_ids)
                    pred_words = pred_text.split()

                    ref_tokens_ids = inp_ids[j].tolist()
                    try:
                        ref_eos_idx = ref_tokens_ids.index(self.eos_token_id)
                        ref_content_ids = ref_tokens_ids[:ref_eos_idx]
                    except ValueError:
                        ref_content_ids = ref_tokens_ids

                    ref_text = self.tokenizer.decode(ref_content_ids)
                    ref_words = ref_text.split()

                    target_lengths.append(tgt_len)
                    actual_lengths.append(actual_len)
                    if actual_len == tgt_len:
                        exact_matches += 1
                    total_samples += 1

                    all_hypotheses.append(pred_words)
                    all_references.append([ref_words])

                    try:
                        m_score = meteor_score([ref_words], pred_words)
                        meteor_scores.append(m_score)
                    except Exception:
                        meteor_scores.append(0.0)

                    r_score = self.calculate_rouge_l(ref_words, pred_words)
                    rouge_scores.append(r_score)

        if total_samples == 0:
            print("WARNING: Evaluator received 0 samples! Returning baseline 0 scores.")
            return {
                "Control_Accuracy": 0.0,
                "Control_MAE": 0.0,
                "BLEU_1": 0.0,
                "BLEU_2": 0.0,
                "BLEU_3": 0.0,
                "BLEU_4": 0.0,
                "METEOR": 0.0,
                "ROUGE_L": 0.0,
                "Count": 0,
            }

        mae = np.mean(np.abs(np.array(actual_lengths) - np.array(target_lengths)))
        accuracy = exact_matches / total_samples

        smoothie = SmoothingFunction().method4
        b1 = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoothie,
        )
        b2 = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothie,
        )
        b3 = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smoothie,
        )
        b4 = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )

        # Generate plot for scores vs target lengths
        self._plot_metrics_vs_length(
            target_lengths,
            all_references,
            all_hypotheses,
            meteor_scores,
            rouge_scores,
            plot_path,
        )

        report = {
            "Control_Accuracy": round(accuracy * 100, 2),
            "Control_MAE": round(mae, 4),
            "BLEU_1": round(b1 * 100, 2),
            "BLEU_2": round(b2 * 100, 2),
            "BLEU_3": round(b3 * 100, 2),
            "BLEU_4": round(b4 * 100, 2),
            "METEOR": round(np.mean(meteor_scores) * 100, 2),
            "ROUGE_L": round(np.mean(rouge_scores) * 100, 2),
            "Count": total_samples,
        }

        return report

    def _plot_metrics_vs_length(
        self,
        target_lengths,
        all_references,
        all_hypotheses,
        meteor_scores,
        rouge_scores,
        save_path,
    ):
        """
        Groups the samples into bins based on their target length and plots
        how BLEU, METEOR, and ROUGE change as output length increases.
        """
        # Bin size of 10 tokens (e.g. 0-9, 10-19, etc.) for smoother visualization
        binned_data: Dict = defaultdict(
            lambda: {"refs": [], "hyps": [], "meteor": [], "rouge": []}
        )

        for i in range(len(target_lengths)):
            bin_key = (
                target_lengths[i] // 10
            ) * 10 + 5  # Use the center of the bin for the X-axis
            binned_data[bin_key]["refs"].append(all_references[i])
            binned_data[bin_key]["hyps"].append(all_hypotheses[i])
            binned_data[bin_key]["meteor"].append(meteor_scores[i])
            binned_data[bin_key]["rouge"].append(rouge_scores[i])

        sorted_bins = sorted(binned_data.keys())
        x_lengths = []
        y_bleu4 = []
        y_meteor = []
        y_rouge = []

        smoothie = SmoothingFunction().method4

        for b in sorted_bins:
            data = binned_data[b]
            # Skip bins with fewer than 5 samples to avoid extreme variance noise
            if len(data["refs"]) < 5:
                continue

            x_lengths.append(b)
            y_meteor.append(np.mean(data["meteor"]) * 100)
            y_rouge.append(np.mean(data["rouge"]) * 100)

            # Calculate corpus BLEU-4 for this specific length bin
            b4 = corpus_bleu(
                data["refs"],
                data["hyps"],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
            y_bleu4.append(b4 * 100)

        if len(x_lengths) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(x_lengths, y_bleu4, marker="o", label="BLEU-4")
            plt.plot(x_lengths, y_meteor, marker="s", label="METEOR")
            plt.plot(x_lengths, y_rouge, marker="^", label="ROUGE-L")

            plt.title("Evaluation Scores vs. Target Output Length")
            plt.xlabel("Target Length (Tokens)")
            plt.ylabel("Score")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Scores vs. Length plot saved to {save_path}")

    def calculate_rouge_l(self, reference, hypothesis):
        if not reference or not hypothesis:
            return 0.0
        m, n = len(reference), len(hypothesis)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        prec = lcs / n if n > 0 else 0
        rec = lcs / m if m > 0 else 0
        if (prec + rec) == 0:
            return 0.0
        f1 = 2 * prec * rec / (prec + rec)
        return f1

    def generate_batch(self, images, target_lens):
        B = images.shape[0]
        input_ids = torch.full(
            (B, 1), self.eos_token_id, dtype=torch.long, device=self.device
        )
        current_lengths = torch.zeros(B, dtype=torch.long, device=self.device)
        finished_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        eos_reached = torch.zeros(B, dtype=torch.bool, device=self.device)

        max_loop = target_lens.max().item() + 5

        while not finished_mask.all():
            if current_lengths.max() > max_loop:
                break

            logits = self.model(images, input_ids, target_lens)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            next_token = torch.where(
                eos_reached, torch.full_like(next_token, self.eos_token_id), next_token
            )
            eos_reached = eos_reached | (next_token == self.eos_token_id)
            next_token = torch.where(
                finished_mask,
                torch.full_like(next_token, self.eos_token_id),
                next_token,
            )

            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            current_lengths += (~finished_mask).long()
            finished_mask = current_lengths >= target_lens

        return input_ids[:, 1:]
