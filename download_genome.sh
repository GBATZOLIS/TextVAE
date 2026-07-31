#!/bin/bash

set -e

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR="./data/visual_genome"
PARQUET_DIR="$DATA_DIR/parquet"
IMAGES_DIR="$DATA_DIR/images"

mkdir -p $DATA_DIR $PARQUET_DIR $IMAGES_DIR

# -----------------------------
# 1. Download Visual Genome Region Descriptions (HuggingFace Parquet)
# -----------------------------
echo "Downloading Visual Genome captions (parquet) from HuggingFace..."
echo "This may take a few minutes and ~15GB disk space..."

hf_repo="ljnlonoljpiljm/visual-genome-region-descriptions"

python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

repo_id = "$hf_repo"
local_dir = "$PARQUET_DIR"
os.makedirs(local_dir, exist_ok=True)

# List of all files in the repo
files = [
    f"data/train-{str(i).zfill(5)}-of-00018.parquet" for i in range(18)
] + ["README.md", ".gitattributes"]

for f in files:
    print("Downloading:", f)
    hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset", local_dir=local_dir, force_download=False)
EOF

echo "Captions downloaded to $PARQUET_DIR"

# -----------------------------
# 2. Move Images safely (if you already have them)
# -----------------------------
echo "Organizing images..."

for PART in VG_100K VG_100K_2; do
    if [ -d "$DATA_DIR/$PART" ]; then
        echo "Moving $PART into $IMAGES_DIR ..."
        find "$DATA_DIR/$PART" -type f -print0 | xargs -0 -I {} mv {} "$IMAGES_DIR/"
        rmdir "$DATA_DIR/$PART" || true
    fi
done

echo "Done! Dataset structure:"
echo " - Captions (parquet): $PARQUET_DIR"
echo " - Images: $IMAGES_DIR"
