import matplotlib

matplotlib.use("Agg")  # Headless backend for HPC/Servers
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def plot_metrics_vs_length(
    target_lengths: List[int],
    all_references: List[List[List[str]]],
    all_hypotheses: List[List[str]],
    meteor_scores: List[float],
    rouge_scores: List[float],
    save_path: str,
):
    """
    Plots the BLEU, METEOR, and ROUGE distributions grouped by target sequence length.
    """
    binned_data: Dict = defaultdict(
        lambda: {"refs": [], "hyps": [], "meteor": [], "rouge": []}
    )

    # Group data into bins of 10 tokens
    for i in range(len(target_lengths)):
        bin_key = (target_lengths[i] // 10) * 10 + 5
        binned_data[bin_key]["refs"].append(all_references[i])
        binned_data[bin_key]["hyps"].append(all_hypotheses[i])
        binned_data[bin_key]["meteor"].append(meteor_scores[i])
        binned_data[bin_key]["rouge"].append(rouge_scores[i])

    sorted_bins = sorted(binned_data.keys())
    x_lengths, y_bleu4, y_meteor, y_rouge = [], [], [], []
    smoothie = SmoothingFunction().method4

    for b in sorted_bins:
        data = binned_data[b]
        if len(data["refs"]) < 1:
            continue
        x_lengths.append(b)
        y_meteor.append(np.mean(data["meteor"]) * 100)
        y_rouge.append(np.mean(data["rouge"]) * 100)

        b4 = corpus_bleu(
            data["refs"],
            data["hyps"],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        y_bleu4.append(b4 * 100)

    if len(x_lengths) > 0:
        plt.figure(figsize=(10, 6))
        fmt = "o" if len(x_lengths) == 1 else "-"

        plt.plot(
            x_lengths,
            y_bleu4,
            marker="o",
            linestyle=fmt if fmt == "o" else "-",
            label="BLEU-4",
        )
        plt.plot(
            x_lengths,
            y_meteor,
            marker="s",
            linestyle=fmt if fmt == "o" else "-",
            label="METEOR",
        )
        plt.plot(
            x_lengths,
            y_rouge,
            marker="^",
            linestyle=fmt if fmt == "o" else "-",
            label="ROUGE-L",
        )

        plt.title("Evaluation Scores vs. Ground Truth Output Length")
        plt.xlabel("Ground Truth Length (Tokens)")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def plot_forced_length_sweep(
    forced_lengths: List[int],
    length_to_metrics: dict,
    has_bert_score: bool,
    save_path: str,
):
    """
    Plots the Semantic & Alignment scores across specific forced target lengths.
    """
    x_vals = forced_lengths
    y_bleu = [length_to_metrics[length]["BLEU_4"] for length in x_vals]
    y_clip = [length_to_metrics[length]["CLIPScore"] for length in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_bleu, marker="o", label="BLEU-4 (Exact Word Matches)")
    plt.plot(x_vals, y_clip, marker="^", label="CLIPScore (Image-Text Alignment)")

    if has_bert_score:
        y_bert = [length_to_metrics[length]["BERTScore_F1"] for length in x_vals]
        plt.plot(x_vals, y_bert, marker="s", label="BERTScore (Semantic Similarity)")

    plt.title("Semantic & Alignment Scores vs. Target Output Length")
    plt.xlabel("Forced Target Length (Tokens)")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"\nSweep Plot successfully saved to: {os.path.abspath(save_path)}")
