# config.py

import torch

# --- Overall Project Config ---
PROJECT_NAME = "Variable_Length_VQVAE"
WANDB_ENTITY = "rg625-university-of-cambridge"  # <-- CHANGE THIS
RUN_NAME = "e2e_run_1"
SEED = 42

# --- Model Config ---
D_MODEL = 128  # The central embedding dimension for all components
MAX_N_GENERATE = 128  # The maximum number of "elaboration" tokens to generate

VQ_VAE_CONFIG = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_residual_layers": 2,
    "residual_hidden_dim": 32,
    "num_embeddings": 512,
    "embedding_dim": D_MODEL,
    "commitment_cost": 0.25,
}

AUTOREGRESSIVE_CONFIG = {
    "n_codes": VQ_VAE_CONFIG["num_embeddings"],
    "d_model": D_MODEL,
    "n_head": 4,
    "n_layers": 4,
    "max_fixed_len": 8 * 8,  # 64, from the VQ-VAE encoder output shape
}

DECODER_CONFIG = {
    "n_pixels_rgb": 256,
    "d_model": D_MODEL,
    "n_head": 4,
    "n_layers": 4,
    # Max seq len must account for the maximum possible latent sequence
    "max_seq_len": (32 * 32 * 3)
    + AUTOREGRESSIVE_CONFIG["max_fixed_len"]
    + MAX_N_GENERATE,
}

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Smaller batch size due to the large model
VALIDATION_BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DATASET_PATH = "~/datasets/cifar10/"

# --- Visualization & Logging ---
LOG_INTERVAL = 100  # Log metrics every 100 batches
SAVE_IMAGE_INTERVAL = 1  # Save a sample image every N epochs
RESULTS_DIR = "results"
