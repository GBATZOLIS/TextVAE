import torch
from .encoder import Encoder
from torchvision import transforms
from PIL import Image
import tiktoken
import argparse
from typing import List


def generate_plan(model, image, target_len, device, tokenizer):
    """
    Generates text while respecting the LDPE countdown.
    """
    model.eval()

    # Preprocess Image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prepare Input
    # Start with empty sequence (or SOS if specific one used)
    # We use an empty list for tiktoken-based generation, typically we rely on visual context first
    # Or start with a specific start token if trained with one.
    input_ids: List = []

    # The crucial component: Target Length Tensor
    # We tell the model: "You must finish in X steps"
    target_len_tensor = torch.tensor([target_len], device=device)

    print(f"--- Generating Plan (Length {target_len}) ---")

    with torch.no_grad():
        for i in range(target_len):
            # Convert current tokens to tensor
            if len(input_ids) == 0:
                # If start, pass dummy or SOS.
                # Here we handle the cold start by passing a dummy token or training with SOS.
                # Assuming training data had no explicit SOS, we might need a dummy start
                # or modify dataset to prepend one.
                # For this SOTA implementation, let's assume we start with 50256 (EOS/SOS)
                curr_input = torch.tensor([[50256]], device=device)
            else:
                curr_input = torch.tensor([input_ids], device=device)

            # Forward
            logits = model(img_tensor, curr_input, target_len_tensor)

            # Greedy decode last token
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            input_ids.append(next_token)

            # Decode for print
            text = tokenizer.decode(input_ids)
            print(f"Step {i+1}/{target_len}: {text}")

            if next_token == 50256:  # EOS
                break

    return tokenizer.decode(input_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--len", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model structure
    model = Encoder(vit_dim=512).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    tokenizer = tiktoken.get_encoding("gpt2")

    image = Image.open(args.img).convert("RGB")

    result = generate_plan(model, image, args.len, device, tokenizer)
    print(f"\nFinal Result: {result}")
