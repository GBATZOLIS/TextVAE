import os
from datasets import load_dataset

# 1. Force Hugging Face to use your high-capacity HPC workspace
cache_dir = "~/datasets/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

print(f"Starting download to: {cache_dir}")
print("This may take a while. Please do not close your terminal...")

# 2. Download the full dataset (Notice we removed streaming=True)
dataset = load_dataset("allenai/PixMo-Cap", split="train", cache_dir=cache_dir)

print(f"\nSuccess! Downloaded {len(dataset)} items to local HPC storage.")
print("Your DDP workers will now read directly from the ultra-fast NVMe/Lustre disks!")
