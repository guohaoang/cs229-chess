"""Configuration settings for the chess ML model."""

# Data
DATASET_DIR = "datasets_small"
DATASET_PATTERN = "lichess_db_standard_rated_*.pgn.zst"
GAMES_PER_FILE = 5000  # Limit for memory (reduced for faster loading)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Board representation
BOARD_SIZE = 8
BOARD_CHANNELS = 13  # 6 piece types * 2 colors + 1 for side to move
MOVE_HISTORY = 0  # Stack past boards for context (0 = current only)

# Model
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZES = [3, 3, 3]
DENSE_LAYERS = [512, 256]
DROPOUT_RATE = 0.5

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0  # Disable for macOS (causes hangs with MPS)

# Device
DEVICE = "cuda"  # or "cpu"
USE_AMP = True  # Automatic Mixed Precision

# Logging
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
LOG_INTERVAL = 100
SAVE_INTERVAL = 5  # epochs
