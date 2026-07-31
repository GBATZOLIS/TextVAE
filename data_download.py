import json
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
# Qwen2.5-7B is incredibly smart, natively supports JSON-like structuring, and easily fits on a 4090
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "allenai/PixMo-Cap"
OUTPUT_FILE = "/home/rg625/datasets/pixmo_stratified.jsonl"
HF_CACHE_DIR = "/home/rg625/datasets/hf_cache"

SYSTEM_PROMPT = """Rewrite the following image caption into three lengths.
Return ONLY a valid JSON object with keys: "short" (5-10 words), "medium" (40-60 words), and "long" (the original details)."""


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=HF_CACHE_DIR)

    # Extract the original captions
    original_captions = [row.get("text", row.get("caption", "")) for row in dataset]
    image_urls = [row.get("image_url", row.get("url", "")) for row in dataset]

    # Format the prompts for the model
    formatted_prompts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{cap}<|im_end|>\n<|im_start|>assistant\n{{"
        for cap in original_captions
        if cap
    ]

    print(f"Loading {MODEL_NAME} into VRAM...")
    # Initialize vLLM (this allocates your GPU memory)
    llm = LLM(model=MODEL_NAME, download_dir=HF_CACHE_DIR, max_model_len=4096)
    sampling_params = SamplingParams(
        temperature=0.7, max_tokens=500, stop=["<|im_end|>"]
    )

    print("Generating augmented captions in batches...")
    # vLLM handles the massive parallel batching automatically
    outputs = llm.generate(formatted_prompts, sampling_params)

    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            try:
                # We forced the prompt to start with '{', so we add it back
                generated_text = "{" + output.outputs[0].text.strip()
                augmented_data = json.loads(generated_text)

                new_row = {
                    "image_url": image_urls[i],
                    "original": original_captions[i],
                    "short": augmented_data.get("short", ""),
                    "medium": augmented_data.get("medium", ""),
                    "long": augmented_data.get("long", original_captions[i]),
                }
                f.write(json.dumps(new_row) + "\n")
            except json.JSONDecodeError:
                # Skip rows where the model failed to output perfect JSON
                continue

    print("Done! You now have a free, locally augmented dataset.")


if __name__ == "__main__":
    main()
