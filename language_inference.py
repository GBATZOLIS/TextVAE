import torch
import argparse

from shapes3d import Shapes3DDataset
from model import ImageToTextModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Dataset (for vocab and transforms) ---
    dataset = Shapes3DDataset(split="train")
    pad_idx = dataset.token_to_idx["[PAD]"]

    # --- Load Model ---
    print(f"Loading model checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Infer model args from checkpoint state_dict
    embed_size = checkpoint["model_state_dict"]["encoder.linear.weight"].shape[0]
    hidden_size = checkpoint["model_state_dict"]["decoder.lstm.weight_hh_l0"].shape[1]

    model = ImageToTextModel(embed_size, hidden_size, len(dataset.token_to_idx)).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Load and Process Image ---
    print(f"Loading data for sample index: {args.sample_index}")
    sample = dataset[args.sample_index]

    image_tensor = sample["image"].unsqueeze(0).to(device)
    ground_truth_tokens = sample["target"].unsqueeze(0).to(device)
    pil_image = sample["pil_image"]

    # --- Get Model Prediction ---
    predicted_string = model.predict(image_tensor, dataset)[0]

    # --- Calculate Log-Likelihood of Ground Truth ---
    log_likelihood = model.calculate_log_likelihood(
        image_tensor, ground_truth_tokens, pad_idx
    ).item()

    # --- Display Results ---
    print("\n--- Inference Results ---")
    try:
        pil_image.show(title=f"Input Image (Index: {args.sample_index})")
    except Exception:
        print(
            "Could not display image. It might be due to running in a non-GUI environment."
        )

    print(f"\nGround Truth: {dataset.detokenize(ground_truth_tokens.squeeze())}")
    print(f"Prediction:   {predicted_string}")
    print(f"\nLog-Likelihood of Ground Truth Sequence: {log_likelihood:.4f}")
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained ImageToText model."
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=100,
        help="Index of the sample from the dataset to test.",
    )
    args = parser.parse_args()
    main(args)
