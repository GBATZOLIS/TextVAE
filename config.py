# config.py

import torch

# --- General ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "~/datasets/cifar10/"
VQVAE_CHECKPOINT_PATH = "vqvae_model.pth"
PRIOR_CHECKPOINT_PATH = "prior_model.pth"  # Checkpoint for the new prior model
LOG_INTERVAL = 100

# --- Data ---
IN_CHANNELS = 3
IMAGE_SIZE = 32

# --- VQ-VAE Architecture ---
PATCH_SIZE = 4
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
EMBEDDING_DIM = 256
ENCODER_LAYERS = 4
ENCODER_HEADS = 8
DECODER_LAYERS = 4
DECODER_HEADS = 8
DROPOUT = 0.1
NUM_EMBEDTINGS = 512
BETA = 0.25

# --- VQ-VAE Training ---
VQVAE_NUM_EPOCHS = 1000
# --- MODIFIED: Reduced batch size to prevent OOM errors ---
VQVAE_BATCH_SIZE = 256
VQVAE_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
APPLY_TRUNCATION = True
PERCEPTUAL_LOSS_WEIGHT = 0.1

# --- Autoregressive Prior Model ---
PRIOR_EMBEDDING_DIM = 256
PRIOR_NHEAD = 8
PRIOR_LAYERS = 6
PRIOR_DROPOUT = 0.1
PRIOR_NUM_EPOCHS = 50
# --- MODIFIED: Reduced batch size for consistency ---
PRIOR_BATCH_SIZE = 64
PRIOR_LEARNING_RATE = 3e-4
