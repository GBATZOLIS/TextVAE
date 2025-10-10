import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Configuration (CIFAR-10) ---
IMG_SIZE = 32
IN_CHANNELS = 3
DATASET_PATH = "~/datasets/cifar10"
# --- Model Configuration ---
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

# --- Model Complexity ---
EMBEDDING_DIM = 128
ENCODER_LAYERS = 4
ENCODER_HEADS = 4
DECODER_LAYERS = 4
DECODER_HEADS = 4
DROPOUT = 0.2

# --- Autoregressive Decoder Specific ---
PIXEL_EMBEDDING_DIM = 128
VOCAB_SIZE = 256  # 256 possible pixel values

# --- VQ-VAE Specific Configuration ---
NUM_EMBEDDINGS = 512
BETA = 1.0

# --- Training Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 2000

# --- Trainer Engine & Logging Configuration ---
LOG_INTERVAL = 100
SAVE_IMAGE_INTERVAL = 100
# The number of pixels to generate autoregressively in the decoder
MAX_N_GENERATE = 512
APPLY_TRUNCATION = True
MIN_SEQ_LEN_FRAC = 0.5

PRIOR_N_BLOCKS = 4
PRIOR_EMBEDDING_DIM = 256
PRIOR_DROPOUT = 0.2
